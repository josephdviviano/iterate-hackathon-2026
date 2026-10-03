"""Networks for the recipe: airbench-lineage nets and a ResNet-9 reference arm.

Every model accepts float32 RGB in [0, 1] (the harness convention), normalises with
per-trial statistics held in buffers, and returns float32 logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig

NUM_CLASSES = 100
WHITEN_KERNEL = 2


def activate(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(x)
    if name == "silu":
        return F.silu(x)
    return F.celu(x, alpha=0.075)  # Page's ResNet-9 CELU


def max_pool2(
    x: torch.Tensor, impl: str, overlap: bool = False, ceil: bool = False
) -> torch.Tensor:
    """2x2 max-pool (floor mode). ``amax`` reshapes and reduces instead of saving indices, so
    its backward is a fusable elementwise mask rather than an atomic scatter. ``overlap`` uses a
    3x3 stride-2 window on odd maps (same output size, no border dropped); ``ceil`` keeps the
    last row and column (ceil mode)."""
    if impl == "torch":
        if overlap and x.size(-1) % 2:
            return F.max_pool2d(x, 3, stride=2)
        return F.max_pool2d(x, 2, ceil_mode=ceil)
    b, c, h, w = x.shape
    x = x[:, :, : h - h % 2, : w - w % 2]
    return x.view(b, c, h // 2, 2, w // 2, 2).amax(dim=(3, 5))


def global_max(x: torch.Tensor, impl: str) -> torch.Tensor:
    if impl == "torch":
        return F.adaptive_max_pool2d(x, 1).flatten(1)
    if impl == "flatmax":
        # Fable's GlobalAmaxPool: max with indices over the flattened map (gather backward).
        return x.flatten(2).max(2).values
    return x.amax(dim=(2, 3))


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
    """Bias-free 3x3 convolution with identity (dirac) initialisation of the first channels.

    ``rep_branch`` adds a parallel trainable 1x1 convolution (RepVGG-style train-time
    over-parameterisation); the sum is a single 3x3 conv at evaluation in principle.
    """

    def __init__(
        self, cin: int, cout: int, kernel_size: int = 3, dirac: bool = True, rep: bool = False
    ) -> None:
        self.dirac = dirac
        super().__init__(cin, cout, kernel_size, padding="same", bias=False)
        self.branch = nn.Conv2d(cin, cout, 1, bias=False) if rep else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return out if self.branch is None else out + self.branch(x)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.dirac:
            w = self.weight.data
            nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    """conv-pool-BN-GELU then one or two conv-BN-GELU; depth 3 adds a residual (airbench96)."""

    def __init__(
        self,
        cin: int,
        cout: int,
        depth: int,
        bn_momentum: float,
        pool_first: bool = False,
        pool_impl: str = "torch",
        activation: str = "gelu",
        rep: bool = False,
        pool_overlap: bool = False,
        pool_ceil: bool = False,
        skip_discarded: bool = False,
        post_add_activation: bool = False,
        skip_gate: bool = False,
    ) -> None:
        super().__init__()
        # SkipInit (De & Smith 2020): a learnable scalar on the residual branch, reset in
        # ``reset_model`` to ``residual_gate_init`` so each block starts near identity.
        self.skip_gate = nn.Parameter(torch.ones(())) if skip_gate else None
        # ResNet-style GELU(residual + x) instead of airbench96's x + GELU(residual).
        self.post_add_activation = post_add_activation
        # Floor-mode 2x2 pooling of an odd map drops the conv's last row and column; pad the
        # top-left only and convolve unpadded so those outputs are never computed (exact).
        self.skip_discarded = skip_discarded and not (pool_overlap or pool_ceil or rep)
        self.pool_overlap = pool_overlap
        self.pool_ceil = pool_ceil
        # Progressive deepening: the residual branch is skipped until activated, then blended
        # in by ``residual_gate`` (a non-persistent 0-dim buffer, so changing it never
        # recompiles and lookahead never averages it).
        self.residual_active = True
        self.residual_ramping = False
        self.register_buffer("residual_gate", torch.ones(()), persistent=False)
        self.pool_impl = pool_impl
        self.activation = activation
        # pool_first max-pools before the widening conv, cutting its cost 4x.
        self.pool_first = pool_first
        self.conv1 = Conv(cin, cout, rep=rep)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout, rep=rep)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.residual = (
            nn.Sequential(Conv(cout, cout, rep=rep), BatchNorm(cout, bn_momentum))
            if depth == 3
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_first:
            x = activate(self.norm1(self.conv1(max_pool2(x, self.pool_impl))), self.activation)
        else:
            if self.skip_discarded and x.size(-1) % 2:
                conv = F.conv2d(F.pad(x, (1, 0, 1, 0)), self.conv1.weight)
            else:
                conv = self.conv1(x)
            pooled = max_pool2(conv, self.pool_impl, self.pool_overlap, self.pool_ceil)
            x = activate(self.norm1(pooled), self.activation)
        y = activate(self.norm2(self.conv2(x)), self.activation)
        if self.residual is None or not self.residual_active:
            return y
        if self.post_add_activation:
            deep = activate(self.residual(y) + x, self.activation)
        elif self.skip_gate is not None:
            deep = x + self.skip_gate * activate(self.residual(y), self.activation)
        else:
            deep = x + activate(self.residual(y), self.activation)
        if not self.residual_ramping:
            return deep
        return torch.lerp(y, deep, self.residual_gate.to(y.dtype))


class AirbenchNet(nn.Module):
    """Whitening conv, three ConvGroups, global max-pool and a scaled bias-free head."""

    def __init__(self, config: RecipeConfig) -> None:
        super().__init__()
        w1, w2, w3 = config.scaled_widths
        k = config.whiten_kernel
        whiten_width = 2 * 3 * k**2
        self.normalize = Normalize()
        self.whiten = nn.Conv2d(3, whiten_width, k, padding=0, bias=True)
        self.activation = config.activation
        self.head_pool = config.head_pool
        self.global_pool = config.global_pool
        self.train_size: int | None = None
        self.whiten.weight.requires_grad = False
        depths = config.stage_depths or (config.block_depth,) * 3
        pools = config.pool_first or (False, False, False)
        momentum = config.bn_momentum
        widths = (whiten_width, w1, w2, w3)
        self.groups = nn.Sequential(
            *(
                ConvGroup(
                    widths[i],
                    widths[i + 1],
                    depths[i],
                    momentum,
                    pools[i],
                    config.pool_impl,
                    config.activation,
                    config.rep_branch,
                    config.pool_overlap,
                    config.stage3_ceil and i == 2,
                    config.skip_discarded,
                    config.post_add_activation,
                    config.residual_gate_init is not None,
                )
                for i in range(3)
            )
        )
        head_in = w3
        # Wide 1x1 expansion on the final map before global pooling (EfficientNet-style head).
        self.expand = None
        if config.head_expand:
            head_in = config.head_expand
            self.expand = nn.Sequential(
                nn.Conv2d(w3, head_in, 1, bias=False), BatchNorm(head_in, config.bn_momentum)
            )
        # Cosine head: logits = s * max over sub-centres of cos(feature, prototype).
        self.cosine_scale = config.cosine_head_scale
        self.subcenters = config.cosine_subcenters
        rows = NUM_CLASSES * (self.subcenters if self.cosine_scale else 1)
        # Fixed simplex-ETF classifier (frozen constant weights) with a learnable class bias.
        self.etf_head = config.etf_head
        self.head = nn.Linear(head_in, rows, bias=config.etf_head)
        if config.etf_head:
            self.head.weight.requires_grad = False
        self.init_gain = config.init_gain
        self.conv_init = config.conv_init
        self.square_init, self.square_beta = config.square_init, config.square_beta
        # Head-feature centring: BatchNorm1d (no affine) on the pooled features, fp32.
        self.center = (
            nn.BatchNorm1d(head_in, affine=False, momentum=1 - config.bn_momentum)
            if config.head_center
            else None
        )
        self.residual_gate_init = config.residual_gate_init
        # Annealed log-sum-exp global pool below the final resolution (``soft_pool`` is set by
        # fit; ``pool_tau`` is a non-persistent buffer so annealing never recompiles).
        self.soft_pool = False
        self.register_buffer("pool_tau", torch.ones(()), persistent=False)
        # Multi-exit: an auxiliary head on stage 2, trained jointly and averaged in at eval.
        self.exit_head = (
            nn.Linear(w2, NUM_CLASSES, bias=False)
            if config.exit_weight or config.exit_eval_weight
            else None
        )
        self.exit_eval_weight = config.exit_eval_weight
        self.pool_impl = config.pool_impl
        self.head_norm = config.head_norm
        self.scale = 1 / w3 if config.head_norm else config.scaling_factor
        # Progressive freezing (FreezeOut-style): the first ``frozen_groups`` stages run
        # without autograd during training, removing their backward cost.
        self.frozen_groups = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = self.train_size if self.training else None
        if size is not None and size != x.size(-1):
            x = F.interpolate(
                x, size=(size, size), mode="bilinear", align_corners=False, antialias=True
            )
        x = self.normalize(x, self.whiten.weight.dtype)
        frozen = self.frozen_groups if self.training else 0
        if frozen:
            with torch.no_grad():
                x = self.groups[:frozen](activate(self.whiten(x), self.activation))
            x = self.groups[frozen:](x)
        elif self.exit_head is not None:
            mid = self.groups[:2](activate(self.whiten(x), self.activation))
            aux = self.exit_head(global_max(mid, self.global_pool or self.pool_impl)) * self.scale
            main = self._head(self.groups[2](mid))
            if self.training:
                return main, aux.float()
            return main + self.exit_eval_weight * aux.float()
        else:
            x = self.groups(activate(self.whiten(x), self.activation))
        return self._head(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate features (for the closed-form head refit)."""
        x = self.normalize(x, self.whiten.weight.dtype)
        x = self.groups(activate(self.whiten(x), self.activation))
        if self.expand is not None:
            x = activate(self.expand(x), self.activation)
        return global_max(x, self.global_pool or self.pool_impl).float()

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand is not None:
            x = activate(self.expand(x), self.activation)
        if self.cosine_scale:
            f = F.normalize(global_max(x, self.global_pool or self.pool_impl).float(), dim=1)
            w = F.normalize(self.head.weight.float(), dim=1)
            cos = (f @ w.T).view(len(f), NUM_CLASSES, self.subcenters).amax(-1)
            return self.cosine_scale * cos
        if self.head_pool == "maxmean":
            # Compile-safe max+mean pooling (adaptive pools, no amax).
            x = 0.5 * (F.adaptive_max_pool2d(x, 1) + F.adaptive_avg_pool2d(x, 1)).flatten(1)
        elif self.soft_pool and self.training:
            flat, tau = x.flatten(2).float(), self.pool_tau.float()
            pooled = tau * (torch.logsumexp(flat / tau, dim=2) - math.log(flat.size(2)))
            x = pooled.to(self.head.weight.dtype)
        else:
            x = global_max(x, self.global_pool or self.pool_impl)
        if self.center is not None:
            x = self.center(x.float()).to(self.head.weight.dtype)
        return (self.head(x) * self.scale).float()


class EnsembleNet(nn.Module):
    """Jointly trained or snapshot members whose single-view logits are averaged at eval."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members]).mean(0)


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


class ConvMixerNet(nn.Module):
    """ConvMixer (Trockman & Kolter): patch embedding, then depthwise spatial mixing with a
    residual and pointwise channel mixing, global average pool and a scaled linear head."""

    def __init__(self, config: RecipeConfig) -> None:
        super().__init__()
        dim, k, p = config.convmixer_dim, config.convmixer_kernel, config.convmixer_patch
        self.normalize = Normalize()
        self.embed = nn.Sequential(nn.Conv2d(3, dim, p, stride=p), nn.GELU(), nn.BatchNorm2d(dim))
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "spatial": nn.Sequential(
                        nn.Conv2d(dim, dim, k, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    ),
                    "channel": nn.Sequential(
                        nn.Conv2d(dim, dim, 1), nn.GELU(), nn.BatchNorm2d(dim)
                    ),
                }
            )
            for _ in range(config.convmixer_depth)
        )
        self.head = nn.Linear(dim, NUM_CLASSES, bias=False)
        self.scale = config.scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(self.normalize(x, self.head.weight.dtype))
        for block in self.blocks:
            x = x + block["spatial"](x)
            x = block["channel"](x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return (self.head(x) * self.scale).float()


def make_model(config: RecipeConfig, device: torch.device) -> nn.Module:
    """Construct the network: fp16 channels_last with fp32 BatchNorm on CUDA, fp32 on CPU."""
    architectures = {"airbench": AirbenchNet, "resnet9": ResNet9, "convmixer": ConvMixerNet}
    model: nn.Module = architectures[config.arch](config)
    model = model.to(device)
    if device.type == "cuda":
        model = model.half().to(memory_format=torch.channels_last)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm | Normalize):
                module.float()
    return model


def dct_basis(size: int) -> torch.Tensor:
    """[size*size, size, size] orthonormal 2-D DCT-II basis patterns."""
    n = torch.arange(size, dtype=torch.float64)
    c = torch.cos(math.pi * (n[None, :] + 0.5) * n[:, None] / size)
    c[0] /= math.sqrt(2)
    c *= math.sqrt(2 / size)
    return torch.einsum("ui,vj->uvij", c, c).reshape(size * size, size, size).float()


def hadamard(n: int) -> torch.Tensor:
    """Sylvester Hadamard matrix of the next power-of-two order >= n."""
    h = torch.ones(1, 1)
    while h.size(0) < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h


@torch.no_grad()
def structured_init_(weight: torch.Tensor, kind: str) -> None:
    """Re-initialise the conv rows after the identity (dirac) block with a structured but
    unlearned construction, matching the default (Kaiming) row norm:

    * ``orthogonal``: orthonormal rows over (in-channel, tap).
    * ``dct``: each filter is a 2-D DCT pattern (cycling through the 9 frequencies) times an
      orthonormal in-channel mixing vector, a Gabor-like frequency filter bank.
    * ``zero``: deterministic ZerO-style (Zhao et al. 2022): Hadamard rows on the centre tap.
    The random draws use the global generator, seeded per trial by the harness."""
    cout, cin, kh, kw = weight.shape
    rest = weight[cin:]
    n = rest.size(0)
    if n == 0:
        return
    device = weight.device
    target = rest.float().flatten(1).norm(dim=1).mean()
    if kind == "orthogonal":
        new = torch.empty(n, cin * kh * kw, device=device)
        nn.init.orthogonal_(new)
        new = new.view(n, cin, kh, kw)
    elif kind == "dct":
        basis = dct_basis(kh).to(device)
        mix = torch.empty(n, cin, device=device)
        nn.init.orthogonal_(mix)
        pattern = basis[torch.arange(n, device=device) % len(basis)]
        new = mix[:, :, None, None] * pattern[:, None]
    else:
        new = torch.zeros(n, cin, kh, kw, device=device)
        new[:, :, kh // 2, kw // 2] = hadamard(max(n, cin))[:n, :cin].to(device)
    new = new / new.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) * target
    rest.copy_(new.to(rest))


@torch.no_grad()
def square_perturb_(weight: torch.Tensor, kind: str, beta: float) -> None:
    """Add ``beta`` times a structured (DCT bank) or random perturbation to the identity rows of
    a dirac-initialised conv, at the default Kaiming row norm (1/sqrt(3)); the rows keep their
    identity path."""
    cout, cin, kh, kw = weight.shape
    rows = min(cout, cin)
    device = weight.device
    if kind == "dirac+dct":
        basis = dct_basis(kh).to(device)
        mix = torch.empty(rows, cin, device=device)
        nn.init.orthogonal_(mix)
        pattern = basis[torch.arange(rows, device=device) % len(basis)]
        new = mix[:, :, None, None] * pattern[:, None]
    else:
        new = torch.randn(rows, cin, kh, kw, device=device)
    new = new / new.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) / math.sqrt(3)
    weight[:rows] += (beta * new).to(weight.dtype)


def simplex_etf(features: int, classes: int) -> torch.Tensor:
    """[classes, features] simplex equiangular tight frame with unit-norm rows, from a fixed
    constant seed (a mathematical constant, identical in every trial)."""
    generator = torch.Generator().manual_seed(0)
    if features < classes:  # no ETF exists; fall back to fixed random unit rows
        return F.normalize(torch.randn(classes, features, generator=generator), dim=1)
    basis, _ = torch.linalg.qr(torch.randn(features, classes, generator=generator))
    centred = torch.eye(classes) - 1 / classes
    return ((classes / (classes - 1)) ** 0.5 * basis @ centred).T


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
        if model.conv_init != "kaiming":
            for module in model.modules():
                if isinstance(module, Conv) and module.dirac:
                    structured_init_(module.weight, model.conv_init)
        if model.square_init != "dirac" and model.square_beta:
            for module in model.modules():
                if isinstance(module, Conv) and module.dirac:
                    square_perturb_(module.weight, model.square_init, model.square_beta)
        if model.residual_gate_init is not None:
            for group in model.groups:
                if group.skip_gate is not None:
                    group.skip_gate.fill_(model.residual_gate_init)
        if model.init_gain != 1:
            for module in model.modules():
                if isinstance(module, Conv):
                    module.weight.mul_(model.init_gain)
        if model.etf_head:
            weight = model.head.weight
            weight.copy_(simplex_etf(weight.size(1), weight.size(0)).to(weight))
            model.head.bias.zero_()
        if model.head_norm:
            model.head.weight.div_(model.head.weight.float().std().to(model.head.weight.dtype))


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
