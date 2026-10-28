"""Transformer LM from scratch (CS336 A1 style): RMSNorm, RoPE, SwiGLU,
causal multi-head attention with selectable sdpa/naive backend.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny": dict(d_model=256, n_layers=4, n_heads=4, ctx=512),
    "small": dict(d_model=512, n_layers=6, n_heads=8, ctx=512),
    "main": dict(d_model=768, n_layers=12, n_heads=12, ctx=512),
}


def _dff(d_model: int) -> int:
    return round(8 / 3 * d_model / 64) * 64


class Linear(nn.Module):
    def __init__(self, in_f: int, out_f: int, std: float = 0.02, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_emb: int, emb_dim: int, std: float = 0.02, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_emb, emb_dim, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float = 10000.0, max_seq: int = 4096):
        super().__init__()
        self.max_seq = max_seq
        self.theta = theta
        self._cos: dict[int, torch.Tensor] = {}
        self._sin: dict[int, torch.Tensor] = {}

    def _build(self, d_head: int, seq: int, device, dtype):
        inv = 1.0 / (self.theta ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
        t = torch.arange(seq, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)  # (seq, d_head/2)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        return cos.to(dtype), sin.to(dtype)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, H, S, D_head)
        d_head = x.shape[-1]
        seq = x.shape[-2] + offset
        if (d_head, seq) not in self._cos or self._cos[(d_head, seq)].device != x.device:
            self._cos[(d_head, seq)], self._sin[(d_head, seq)] = self._build(
                d_head, seq, x.device, torch.float32
            )
        cos = self._cos[(d_head, seq)][offset: offset + x.shape[-2]].to(x.dtype)
        sin = self._sin[(d_head, seq)][offset: offset + x.shape[-2]].to(x.dtype)
        return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn: str = "sdpa"):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn = attn
        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.o = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        q = rope(q)
        k = rope(k)
        if self.attn == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attn == "naive":
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            out = torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v
        else:
            raise ValueError(self.attn)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn: str):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, attn)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 ctx: int = 512, d_ff: int | None = None, attn: str = "sdpa",
                 tie_embeddings: bool = False, residual_init_scale: bool = False):
        super().__init__()
        d_ff = d_ff or _dff(d_model)
        self.ctx = ctx
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, ctx=ctx, d_ff=d_ff, attn=attn,
                           tie_embeddings=tie_embeddings,
                           residual_init_scale=residual_init_scale)
        self.tok_emb = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(max_seq=max(ctx * 2, 4096))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, attn) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if residual_init_scale:
            # GPT-2 / nanoGPT trick: shrink residual-branch output projections
            # so block outputs start at unit-ish variance regardless of depth.
            scale = 0.02 / math.sqrt(2 * n_layers)
            with torch.no_grad():
                for block in self.blocks:
                    block.attn.o.weight.normal_(0.0, scale)
                    block.mlp.w2.weight.normal_(0.0, scale)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.lm_head(self.norm_f(x))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config["tie_embeddings"]:
                n -= self.lm_head.weight.numel()
        return n


def build_model(preset: str, vocab_size: int, attn: str = "sdpa") -> TransformerLM:
    cfg = PRESETS[preset]
    return TransformerLM(vocab_size=vocab_size, attn=attn, **cfg)
