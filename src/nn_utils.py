"""From-scratch training utilities: cross-entropy, AdamW, cosine schedule,
gradient clipping. Validated against torch reference implementations in tests.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean CE over flattened (N, V) logits and (N,) int targets."""
    logits = logits.float()
    m = logits.max(dim=-1, keepdim=True).values
    lse = (logits - m).exp().sum(dim=-1, keepdim=True).log() + m
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse.squeeze(-1) - picked).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total = torch.sqrt(sum(torch.linalg.norm(p.grad.detach()) ** 2 for p in params))
    if total > max_norm:
        scale = max_norm / (total + eps)
        for p in params:
            p.grad.detach().mul_(scale)


def cosine_lr_schedule(
    it: int, alpha_max: float, alpha_min: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """Linear warmup to alpha_max, cosine decay to alpha_min at cosine_cycle_iters."""
    if it < warmup_iters:
        return alpha_max * it / warmup_iters
    if it <= cosine_cycle_iters:
        t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * t))
    return alpha_min


class AdamW:
    """AdamW with decoupled weight decay, torch-optim-compatible update order."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        # exact torch.optim.AdamW op order for bit-comparable results
        step_size = self.lr / bc1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (v / bc2).sqrt().add_(self.eps)
            p.mul_(1 - self.lr * self.wd)
            p.addcdiv_(m, denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        for dst, src in zip(self.m, sd["m"]):
            dst.copy_(src)
        for dst, src in zip(self.v, sd["v"]):
            dst.copy_(src)
