"""Faithful CIFAR-100 port of hiverge's cifar10-speedrun record (comparison arm only).

Source: https://github.com/hiverge/cifar10-speedrun (cifar10_speedrun.py, commit 06c6072, MIT
licence). The Newton-Schulz kernel and the vectorised Muon optimiser are copied verbatim; the
network (64/256/256 SiLU CifarNet), augmentation (pre-flip + alternating flip, vectorised
translate 2, brightness/contrast jitter), optimiser split, schedules and hyperparameters follow
the original. Changes required by this harness and its rules: 100-class head; normalisation
statistics computed per trial in prepare (not hard-coded constants); single-view evaluation (the
original's selective test-time augmentation is banned); no sleeps between trials; the
build/prepare/train split with a synthetic warm-up in build. Widths and epochs are parameters so
the method can be run at larger budgets.
"""

from __future__ import annotations

from math import ceil
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = {"widths": [64, 256, 256], "epochs": 7.65, "batch_size": 1536}


@torch.compile(fullgraph=True)
def _zeropower_via_newtonschulz5(
    gradients_4d: list[torch.half],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    current_step: int,
    total_steps: int,
) -> list[torch.half]:
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-05
    eps_gms = 1e-05
    progress_ratio = current_step / max(1, total_steps)

    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = (
        initial_target_mag * (1 - progress_ratio) + final_target_mag * progress_ratio
    )

    # Use stack instead of pre-allocated tensor for better performance
    if not filter_meta_data:
        return gradients_4d

    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        grad_to_orthogonalize = gradients_4d[list_idx]
        g_reshaped = grad_to_orthogonalize.reshape(reshaped_D, reshaped_K)
        padding_dims = (0, max_K - reshaped_K, 0, max_D - reshaped_D)
        g_padded = F.pad(g_reshaped, padding_dims, "constant", 0)
        grad_list.append(g_padded)

    if not grad_list:
        return gradients_4d

    X = torch.stack(grad_list)
    
    # Fuse normalization operations for better performance
    current_batch_mags = X.norm(dim=(1, 2), keepdim=True)
    scale_factor = target_magnitude / (current_batch_mags + eps_gms)
    X = X * scale_factor
    
    X_norm = X.norm(dim=(1, 2), keepdim=True)
    X = X / (X_norm + eps_stable)
    
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True
    
    # Unroll the loop for better performance
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    if transposed:
        X = X.transpose(1, 2)
        
    final_orthogonalized_grads_list = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        orthogonalized_g_padded = X[i]
        orthogonalized_g_reshaped = orthogonalized_g_padded[:reshaped_D, :reshaped_K]
        final_orthogonalized_grads_list[list_idx] = orthogonalized_g_reshaped.view(
            original_shape
        )
    return final_orthogonalized_grads_list


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.08,
        momentum=0.88,
        nesterov=True,
        norm_freq=1,
        total_train_steps=None,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps
        self.filter_params_meta = []
        self.max_D, self.max_K = (0, 0)
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // p.shape[0]
                    self.filter_params_meta.append(
                        {
                            "param": p,
                            "original_shape": p.data.shape,
                            "reshaped_dims": (reshaped_D, reshaped_K),
                        }
                    )
                    self.max_D = max(self.max_D, reshaped_D)
                    self.max_K = max(self.max_K, reshaped_K)
        self.max_D = max(1, self.max_D)
        self.max_K = (
            (max(1, self.max_K) + 15) // 16 * 16
        )
        self.current_grad_norms = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        progress = self.step_count / self.total_train_steps
        group["norm_freq"] = 2 + int(15 * progress)
        # Prepare momentum buffers and track meta data
        filter_params_with_grad = []
        filter_meta_for_current_step = []
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None

        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is not None:
                filter_params_with_grad.append(p)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad,
                        dtype=group["momentum_buffer_dtype"],
                        memory_format=torch.preserve_format)
                if momentum_buffers is not None:
                    momentum_buffers.append(state["momentum_buffer"])
                filter_meta_for_current_step.append((
                    p_meta["original_shape"],
                    p_meta["reshaped_dims"][0],
                    p_meta["reshaped_dims"][1],
                    len(filter_params_with_grad) - 1  # Index in filter_params_with_grad
                ))

        if not filter_params_with_grad:
            return

        # Apply momentum and add gradients
        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [g.to(mb.dtype) for g, mb in zip([p.grad for p in filter_params_with_grad], momentum_buffers)]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]

        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad], momentum_buffers, alpha=group["momentum"])
        else:
            nesterov_grads = momentum_buffers

        do_norm_scaling = (self.step_count - self.last_norm_step >= group["norm_freq"])
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            self.current_grad_norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [
                (len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                for p, n in zip(filter_params_with_grad, self.current_grad_norms)]

        final_orthogonalized_grads = _zeropower_via_newtonschulz5(
            nesterov_grads,
            filter_meta_for_current_step,
            self.max_D,
            self.max_K,
            self.step_count,
            self.total_train_steps,
        )

        # Apply updates in a single fused operation when possible
        if do_norm_scaling:
            # Scale gradients first
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
            # Then apply the orthogonalized updates
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"])
        else:
            # Apply optimizer step directly
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"])

        # Apply weight decay in a fused operation
        weight_decay_factor = 1 - group["lr"] * group["weight_decay"]
        if weight_decay_factor != 1.0:
            torch._foreach_mul_(filter_params_with_grad, weight_decay_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


def batch_color_jitter(inputs, brightness_range: float, contrast_range: float):
    b = inputs.shape[0]
    shift = (torch.rand(b, 1, 1, 1, device=inputs.device, dtype=inputs.dtype) * 2 - 1) * brightness_range
    scale = (torch.rand(b, 1, 1, 1, device=inputs.device, dtype=inputs.dtype) * 2 - 1) * contrast_range + 1
    return (inputs + shift) * scale


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    b, c, h_padded, _ = images.shape
    r = (h_padded - crop_size) // 2
    y_off = (torch.rand(b, device=images.device) * (2 * r + 1)).long()
    x_off = (torch.rand(b, device=images.device) * (2 * r + 1)).long()
    ys = (y_off.view(b, 1, 1, 1) + torch.arange(crop_size, device=images.device).view(1, 1, crop_size, 1)).expand(b, c, crop_size, crop_size)
    xs = (x_off.view(b, 1, 1, 1) + torch.arange(crop_size, device=images.device).view(1, 1, 1, crop_size)).expand(b, c, crop_size, crop_size)
    bi = torch.arange(b, device=images.device).view(b, 1, 1, 1).expand_as(ys)
    ci = torch.arange(c, device=images.device).view(1, c, 1, 1).expand_as(ys)
    return images[bi, ci, ys, xs]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1, self.pool, self.norm1 = Conv(cin, cout), nn.MaxPool2d(2), BatchNorm(cout)
        self.conv2, self.norm2, self.activ = Conv(cout, cout), BatchNorm(cout), nn.SiLU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self, widths, num_classes):
        super().__init__()
        whiten_width = 2 * 3 * 2**2
        self.whiten = nn.Conv2d(3, whiten_width, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0]),
            ConvGroup(widths[0], widths[1]),
            ConvGroup(widths[1], widths[2]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def reset(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()
        w = self.head.weight.data
        w.mul_(1.0 / w.std())
        self.whiten.bias.data.zero_()

    @torch.no_grad()
    def init_whiten(self, train_images, eps=0.0005):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        flat = patches.view(len(patches), -1)
        cov = flat.t() @ flat / len(flat)
        u, s, _ = torch.svd(cov)
        scaled = (u * torch.rsqrt(s + eps).unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat((scaled, -scaled)).to(self.whiten.weight.dtype)

    def forward(self, x, whiten_bias_grad=True):
        if x.dtype != self.whiten.weight.dtype:  # evaluation: float [0, 1] from the harness
            x = ((x - self.mean) / self.std).to(self.whiten.weight.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return (self.head(x) / x.size(-1)).float()


def build(context: BuildContext) -> SimpleNamespace:
    hyp = {**DEFAULTS, **context.parameters}
    device = context.device
    torch.backends.cudnn.benchmark = True
    model = CifarNet(hyp["widths"], context.num_classes).to(device)
    for mod in model.modules():
        mod.half()
    model.mean.data = model.mean.float()
    model.std.data = model.std.float()
    model.to(memory_format=torch.channels_last)

    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        return F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")

    cuda = device.type == "cuda"
    step = torch.compile(forward_step, mode="max-autotune", fullgraph=True) if cuda else forward_step
    state = SimpleNamespace(hyp=hyp, device=device, model=model, forward_step=step)
    if cuda:
        n = 50_000
        synthetic = TrainingData(
            images=torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
            labels=torch.randint(0, context.num_classes, (n,)),
        )
        prepare(state, synthetic, 0)
        train(state)
        model.eval()
        with torch.inference_mode():
            for b in (context.eval_batch_size, 10_000 % context.eval_batch_size or context.eval_batch_size):
                model(torch.rand(b, 3, 32, 32, device=device))
        torch.cuda.synchronize(device)
    return state


def prepare(state: SimpleNamespace, data: TrainingData, seed: int) -> None:
    model, hyp, device = state.model, state.hyp, state.device
    model.reset()
    images = data.images.to(device, non_blocking=True).half().div_(255)
    std, mean = torch.std_mean(images.float(), dim=(0, 2, 3))
    model.mean.copy_(mean.view(1, 3, 1, 1))
    model.std.copy_(std.view(1, 3, 1, 1))
    norm = ((images - mean.view(1, 3, 1, 1).half()) / std.view(1, 3, 1, 1).half()).contiguous(
        memory_format=torch.channels_last
    )
    model.init_whiten(norm[:960])
    flipped = batch_flip_lr(norm)
    state.padded = F.pad(flipped, (2,) * 4, "reflect")
    state.labels = data.labels.to(device, non_blocking=True)
    state.size = images.shape[-1]


def train(state: SimpleNamespace) -> nn.Module:
    model, hyp = state.model, state.hyp
    bs = hyp["batch_size"]
    steps_per_epoch = len(state.labels) // bs
    total_train_steps = ceil(hyp["epochs"] * steps_per_epoch)
    whiten_bias_train_steps = ceil(0.2 * steps_per_epoch)
    bias_lr, head_lr = 0.0573, 0.5415
    wd = 1.0418e-06 * bs
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    opt1 = torch.optim.SGD(
        [
            dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ],
        momentum=0.825,
        nesterov=True,
        fused=state.device.type == "cuda",
    )
    opt2 = Muon(filter_params, lr=0.205, momentum=0.655, nesterov=True, norm_freq=4,
                total_train_steps=total_train_steps, weight_decay=wd)
    opt2.param_groups[0]["momentum_buffer_dtype"] = torch.half
    optimizers = [opt1, opt2]
    initial = [[g["lr"] for g in o.param_groups] for o in optimizers]
    model.train()
    step = 0
    for epoch in range(ceil(total_train_steps / steps_per_epoch)):
        images = batch_crop(state.padded, state.size)
        if epoch % 2 == 1:
            images = images.flip(-1)
        images = batch_color_jitter(images, 0.1399, 0.1308)
        order = torch.randperm(len(state.labels), device=state.labels.device)
        for i in range(steps_per_epoch):
            idx = order[i * bs : (i + 1) * bs]
            loss = state.forward_step(images[idx], state.labels[idx], step < whiten_bias_train_steps)
            loss.backward()
            f1 = 1 - step / max(1, whiten_bias_train_steps)
            f2 = 1 - step / total_train_steps
            opt1.param_groups[0]["lr"] = initial[0][0] * f1
            for g, lr0 in zip(opt1.param_groups[1:], initial[0][1:]):
                g["lr"] = lr0 * f2
            opt2.param_groups[0]["lr"] = initial[1][0] * f2
            for o in optimizers:
                o.step()
                o.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        if step >= total_train_steps:
            break
    del state.padded
    return model
