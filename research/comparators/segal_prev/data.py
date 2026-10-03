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


class TrainingStream:
    """Per-trial training data on the device; yields shuffled, augmented, fixed-size batches.

    Alternating flip: a random flip per image is fixed once, then every image is flipped on
    odd epochs, so each image is seen in both orientations on consecutive epochs.
    """

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
        flip = torch.rand(len(images), device=images.device, generator=generator) < 0.5
        base = torch.where(flip.view(-1, 1, 1, 1), images.flip(-1), images)
        pad = config.translate
        self.base = F.pad(base, (pad,) * 4, mode="reflect") if pad else base
        self.size = images.size(-1)
        self.steps_per_epoch = len(images) // config.batch_size

    def epoch(self, index: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        images = self.base
        if self.config.translate:
            images = batch_crop(images, self.size, self.generator)
        if index % 2 == 1:
            images = images.flip(-1)
        order = torch.randperm(len(images), device=images.device, generator=self.generator)
        bs = self.config.batch_size
        for step in range(self.steps_per_epoch):
            idx = order[step * bs : (step + 1) * bs]
            yield images[idx], self.labels[idx]
