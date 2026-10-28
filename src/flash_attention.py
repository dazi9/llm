"""Causal FlashAttention forward in Triton (online softmax), with a naive
PyTorch reference. Forward-only: the training path uses F.scaled_dot_product_attention
(cuDNN/FlashAttention backend); this kernel exists to verify we understand the
algorithm and to benchmark forward attention in isolation.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, N_CTX,
    sm_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh

    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * sm_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32)  # (BLOCK_M, BLOCK_N)
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        s = tl.where((offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX), s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        # exp2(x*log2e) == exp(x); guard m_new == -inf rows (fully masked).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.math.exp2((s - m_safe[:, None]) * 1.4426950408889634)
        p = tl.where(s == float("-inf"), 0.0, p)
        alpha = tl.math.exp2((m_i - m_safe) * 1.4426950408889634)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True, sm_scale: float | None = None) -> torch.Tensor:
    """Reference (B, H, S, D) attention with explicit masking."""
    B, H, S, D = q.shape
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    scores = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    if causal:
        mask = torch.ones(S, S, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out.to(v.dtype)


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     causal: bool = True, sm_scale: float | None = None,
                     block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """FlashAttention forward, q/k/v: (B, H, S, D), D must be a power of two."""
    B, H, S, D = q.shape
    assert D in (16, 32, 64, 128), f"unsupported head dim {D}"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, block_m), B * H)
    _flash_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, S, sm_scale,
        D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
        num_warps=4,
    )
    return o
