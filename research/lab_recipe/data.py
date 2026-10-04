"""Device-resident training data with airbench-style batched augmentation (all timed work)."""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import torch
import torch.nn.functional as F

from .config import RecipeConfig


def batch_crop(images: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
    """Random translation: crop ``size`` windows from pre-padded images, one mask per shift."""
    r = (images.size(-1) - size) // 2
    shifts = torch.randint(-r, r + 1, (len(images), 2), device=images.device, generator=generator)
    out = torch.empty((len(images), 3, size, size), device=images.device, dtype=images.dtype)
    rows = torch.empty(
        (len(images), 3, size, size + 2 * r), device=images.device, dtype=images.dtype
    )
    for s in range(-r, r + 1):
        mask = shifts[:, 0] == s
        rows[mask] = images[mask, :, r + s : r + s + size, :]
    for s in range(-r, r + 1):
        mask = shifts[:, 1] == s
        out[mask] = rows[mask, :, :, r + s : r + s + size]
    return out


def batch_cutout(
    images: torch.Tensor, size: int, fill: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Replace one random ``size`` square per image with the per-channel training mean."""
    n, _, h, w = images.shape
    centre = torch.randint(0, h, (n, 2), device=images.device, generator=generator)
    ys = torch.arange(h, device=images.device).view(1, h, 1)
    xs = torch.arange(w, device=images.device).view(1, 1, w)
    top = (centre[:, 0] - size // 2).view(n, 1, 1)
    left = (centre[:, 1] - size // 2).view(n, 1, 1)
    mask = (ys >= top) & (ys < top + size) & (xs >= left) & (xs < left + size)
    return torch.where(mask.unsqueeze(1), fill.view(1, 3, 1, 1).to(images.dtype), images)


def colour_jitter(
    images: torch.Tensor, brightness: float, contrast: float, generator: torch.Generator
) -> torch.Tensor:
    """Per-image random brightness scale and contrast about the image mean (hiverge-style)."""
    n = len(images)
    u = torch.rand(n, 2, device=images.device, generator=generator) * 2 - 1
    b = (1 + brightness * u[:, 0]).view(n, 1, 1, 1).to(images.dtype)
    c = (1 + contrast * u[:, 1]).view(n, 1, 1, 1).to(images.dtype)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return ((images - mean) * c + mean * b).clamp_(0, 1)


def hiverge_jitter(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    stats: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """hiverge jitter in normalised units: x_n += U(-b, b), then x_n *= 1 + U(-c, c)."""
    std, mean = (t.view(1, 3, 1, 1).to(images.dtype) for t in stats)
    n = len(images)
    u = torch.rand(n, 2, device=images.device, generator=generator) * 2 - 1
    shift = (brightness * u[:, 0]).view(n, 1, 1, 1).to(images.dtype)
    scale = (1 + contrast * u[:, 1]).view(n, 1, 1, 1).to(images.dtype)
    normalised = ((images - mean) / std + shift) * scale
    return normalised * std + mean


class TrainingStream:
    """Per-trial training data on the device; yields shuffled, augmented, fixed-size batches."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        config: RecipeConfig,
        generator: torch.Generator,
    ) -> None:
        if config.batch_size > len(images):
            raise ValueError(
                f"batch_size {config.batch_size} exceeds the {len(images)} training images"
            )
        self.config = config
        self.generator = generator
        self.labels = labels
        self.channel_mean = images.float().mean(dim=(0, 2, 3))
        self.stats = torch.std_mean(images.float(), dim=(0, 2, 3))
        base = images
        if config.flip != "none":
            # Alternating flip fixes a random flip per image in epoch 0, then flips every image
            # on odd epochs; random flip redraws the mask every epoch.
            base = self._random_flip(images)
        pad = config.translate
        self.base = F.pad(base, (pad,) * 4, mode="reflect") if pad else base
        self.size = images.size(-1)
        self.steps_per_epoch = len(images) // config.batch_size
        self.count = len(images)
        # Per-image scores (loss or gradient-norm proxy) from each image's latest visit.
        self.scores = torch.zeros(len(images), device=images.device) if config.prune_frac else None

    def _balanced_order(self) -> torch.Tensor:
        """Class-interleaved order: every run of C consecutive examples holds one per class,
        so each batch is close to class-balanced (falls back to random if classes differ)."""
        labels = self.labels
        counts = torch.bincount(labels)
        if (counts != counts[0]).any():
            return torch.randperm(len(labels), device=labels.device, generator=self.generator)
        noise = torch.rand(len(labels), device=labels.device, generator=self.generator)
        by_class = torch.argsort(labels.float() + noise * 0.5).view(len(counts), -1)
        rounds = by_class.T  # [per_class, classes]
        shuffle = torch.argsort(
            torch.rand(rounds.shape, device=labels.device, generator=self.generator), dim=1
        )
        return torch.gather(rounds, 1, shuffle).reshape(-1)

    def batch_size(self, index: int) -> int:
        """Batch size of epoch ``index`` under ``batch_schedule`` ((start_epoch, size), ...)."""
        size = self.config.batch_size
        for start, entry in self.config.batch_schedule:
            if index >= start:
                size = entry
        return size

    def steps_until(self, epochs: float) -> int:
        """Training steps in the first ``epochs`` epochs (a fractional last epoch rounds up)."""
        if not self.config.batch_schedule:
            return ceil(self.steps_per_epoch * epochs)
        whole = int(epochs)
        steps = sum(self.count // self.batch_size(e) for e in range(whole))
        if epochs > whole:
            steps += ceil((self.count // self.batch_size(whole)) * (epochs - whole))
        return steps

    def _kept(self) -> torch.Tensor:
        """Indices kept this epoch: drop ``prune_frac`` of images by banked score (lowest for
        ``easy``, highest for ``hard``, half each for ``split``), or for ``soft`` sample the kept
        set without replacement with probability proportional to score**alpha."""
        config, scores = self.config, self.scores
        n = len(scores)
        drop = int(config.prune_frac * n)
        if config.prune_mode == "soft":
            weights = (scores.clamp_min(1e-6) ** config.prune_alpha).float()
            return torch.multinomial(weights, n - drop, generator=self.generator)
        ranked = torch.argsort(scores)  # ascending: easiest first
        if config.prune_mode == "easy":
            return ranked[drop:]
        if config.prune_mode == "hard":
            return ranked[: n - drop]
        return ranked[drop // 2 : n - (drop - drop // 2)]

    @torch.no_grad()
    def observe(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Bank each image's score from the logits already computed for its training step."""
        logits = outputs.float()
        if self.config.prune_score == "loss":
            score = F.cross_entropy(logits, labels, reduction="none")
        else:  # norm of the logit gradient of cross-entropy, ||softmax - onehot||
            score = (logits.softmax(1) - F.one_hot(labels, logits.size(1))).norm(dim=1)
        self.scores[self.last_idx] = score

    def _random_flip(self, images: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(len(images), device=images.device, generator=self.generator) < 0.5
        return torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)

    def epoch(self, index: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        config = self.config
        images = self.base
        clean = config.clean_tail_epochs is not None and index >= config.clean_tail_epochs
        if config.translate and clean:
            r = config.translate  # clean tail: centre crop, no translation
            images = images[:, :, r:-r, r:-r]
        elif config.translate:
            images = batch_crop(images, self.size, self.generator)
        if config.flip == "alternating" and index % 2 == 1:
            images = images.flip(-1)
        elif config.flip == "random" and index > 0:
            images = self._random_flip(images)
        if config.cutout:
            images = batch_cutout(images, config.cutout, self.channel_mean, self.generator)
        if config.brightness or config.contrast:
            if config.jitter_mode == "hiverge":
                images = hiverge_jitter(
                    images, config.brightness, config.contrast, self.stats, self.generator
                )
            else:
                images = colour_jitter(images, config.brightness, config.contrast, self.generator)
        if config.order == "balanced":
            order = self._balanced_order()
        else:
            order = torch.randperm(len(images), device=images.device, generator=self.generator)
        if self.scores is not None and index >= config.prune_start:
            keep = self._kept()
            order = keep[torch.randperm(len(keep), device=keep.device, generator=self.generator)]
        bs = self.batch_size(index)
        for step in range(len(order) // bs):
            idx = order[step * bs : (step + 1) * bs]
            self.last_idx = idx
            yield images[idx], self.labels[idx]
