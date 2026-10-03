"""Device-resident training data with airbench-style batched augmentation (all timed work)."""

from __future__ import annotations

from collections.abc import Iterator

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
        base = images
        if config.flip != "none":
            # Alternating flip fixes a random flip per image in epoch 0, then flips every image
            # on odd epochs; random flip redraws the mask every epoch.
            base = self._random_flip(images)
        pad = config.translate
        self.base = F.pad(base, (pad,) * 4, mode="reflect") if pad else base
        self.size = images.size(-1)
        self.steps_per_epoch = len(images) // config.batch_size

    def _random_flip(self, images: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(len(images), device=images.device, generator=self.generator) < 0.5
        return torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)

    def epoch(self, index: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        config = self.config
        images = self.base
        if config.translate:
            images = batch_crop(images, self.size, self.generator)
        if config.flip == "alternating" and index % 2 == 1:
            images = images.flip(-1)
        elif config.flip == "random" and index > 0:
            images = self._random_flip(images)
        if config.cutout:
            images = batch_cutout(images, config.cutout, self.channel_mean, self.generator)
        order = torch.randperm(len(images), device=images.device, generator=self.generator)
        bs = config.batch_size
        for step in range(self.steps_per_epoch):
            idx = order[step * bs : (step + 1) * bs]
            yield images[idx], self.labels[idx]
