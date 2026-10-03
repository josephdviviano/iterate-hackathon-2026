"""Networks for the recipe: airbench-lineage nets and a ResNet-9 reference arm.

Every model accepts float32 RGB in [0, 1] (the harness convention), normalises with
per-trial statistics held in buffers, and returns float32 logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig

NUM_CLASSES = 100
WHITEN_KERNEL = 2


class Normalize(nn.Module):
    """Per-channel normalisation whose statistics are computed from training data in prepare."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        return x.to(dtype).contiguous(memory_format=torch.channels_last)


class BatchNorm(nn.BatchNorm2d):
    """airbench BatchNorm: frozen unit scale, learnable bias, momentum in airbench convention."""

    def __init__(self, channels: int, momentum: float, trainable_weight: bool = False) -> None:
        super().__init__(channels, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = trainable_weight


class Conv(nn.Conv2d):
    """Bias-free 3x3 convolution with identity (dirac) initialisation of the first channels."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, dirac: bool = True) -> None:
        self.dirac = dirac
        super().__init__(cin, cout, kernel_size, padding="same", bias=False)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.dirac:
            w = self.weight.data
            nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    """conv-pool-BN-GELU then one or two conv-BN-GELU; depth 3 adds a residual (airbench96)."""

    def __init__(self, cin: int, cout: int, depth: int, bn_momentum: float) -> None:
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.residual = (
            nn.Sequential(Conv(cout, cout), BatchNorm(cout, bn_momentum)) if depth == 3 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(F.max_pool2d(self.conv1(x), 2)))
        y = F.gelu(self.norm2(self.conv2(x)))
        if self.residual is None:
            return y
        return x + F.gelu(self.residual(y))


class AirbenchNet(nn.Module):
    """Whitening conv, three ConvGroups, global max-pool and a scaled bias-free head."""

    def __init__(self, config: RecipeConfig) -> None:
        super().__init__()
        w1, w2, w3 = config.scaled_widths
        whiten_width = 2 * 3 * WHITEN_KERNEL**2
        self.normalize = Normalize()
        self.whiten = nn.Conv2d(3, whiten_width, WHITEN_KERNEL, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        depth, momentum = config.block_depth, config.bn_momentum
        self.groups = nn.Sequential(
            ConvGroup(whiten_width, w1, depth, momentum),
            ConvGroup(w1, w2, depth, momentum),
            ConvGroup(w2, w3, depth, momentum),
        )
        self.head = nn.Linear(w3, NUM_CLASSES, bias=False)
        self.scale = config.scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x, self.whiten.weight.dtype)
        x = self.groups(F.gelu(self.whiten(x)))
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return (self.head(x) * self.scale).float()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, bn_momentum: float) -> None:
        super().__init__()
        self.conv1 = Conv(channels, channels, dirac=False)
        self.norm1 = BatchNorm(channels, bn_momentum, trainable_weight=True)
        self.conv2 = Conv(channels, channels, dirac=False)
        self.norm2 = BatchNorm(channels, bn_momentum, trainable_weight=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        return x + F.relu(self.norm2(self.conv2(y)))


class ResNet9(nn.Module):
    """Page-style ResNet-9 reference arm without the airbench initialisation tricks."""

    def __init__(self, config: RecipeConfig) -> None:
        super().__init__()
        m = config.width_mult
        c0, c1, c2, c3 = (max(8, int(round(c * m / 8)) * 8) for c in (64, 128, 256, 512))
        momentum = config.bn_momentum
        self.normalize = Normalize()

        def stage(cin: int, cout: int, pool: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                Conv(cin, cout, dirac=False),
                BatchNorm(cout, momentum, trainable_weight=True),
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.prep = stage(3, c0, pool=False)
        self.body = nn.Sequential(
            stage(c0, c1, pool=True),
            ResidualBlock(c1, momentum),
            stage(c1, c2, pool=True),
            stage(c2, c3, pool=True),
            ResidualBlock(c3, momentum),
        )
        self.head = nn.Linear(c3, NUM_CLASSES, bias=False)
        self.scale = config.scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x, self.prep[0].weight.dtype)
        x = self.body(self.prep(x))
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return (self.head(x) * self.scale).float()


def make_model(config: RecipeConfig, device: torch.device) -> nn.Module:
    """Construct the network: fp16 channels_last with fp32 BatchNorm on CUDA, fp32 on CPU."""
    model: nn.Module = AirbenchNet(config) if config.arch == "airbench" else ResNet9(config)
    model = model.to(device)
    if device.type == "cuda":
        model = model.half().to(memory_format=torch.channels_last)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d | Normalize):
                module.float()
    return model


@torch.no_grad()
def reset_model(model: nn.Module) -> None:
    """Reinitialise every parameter and buffer in place, keeping tensor identities."""
    for module in model.modules():
        if isinstance(module, Normalize):
            module.mean.zero_()
            module.std.fill_(1.0)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
    if isinstance(model, AirbenchNet):
        model.whiten.bias.zero_()


@torch.no_grad()
def init_whitening(model: AirbenchNet, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen whitening conv to +/- scaled eigenpatches of normalised training images."""
    layer = model.whiten
    x = model.normalize(images, torch.float32)
    c, (h, w) = x.shape[1], layer.weight.shape[2:]
    patches = x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c * h * w).float()
    covariance = patches.T @ patches / len(patches)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.copy_(torch.cat((scaled, -scaled)).to(layer.weight.dtype))
