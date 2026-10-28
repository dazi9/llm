import pytest
import torch

from flash_attention import naive_attention, triton_attention

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

HAS_CUDA = torch.cuda.is_available() and HAS_TRITON


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_naive(dtype):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = triton_attention(q, k, v)
    ref = naive_attention(q, k, v)
    assert out.shape == ref.shape
    assert not torch.isnan(out.float()).any()
    tol = 5e-3 if dtype == torch.float32 else 3e-2  # tl.dot uses TF32 for fp32 inputs
    err = (out.float() - ref.float()).abs().max().item()
    assert err < tol, f"max err {err}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA + triton")
def test_triton_short_sequence():
    q = torch.randn(1, 2, 37, 64, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = triton_attention(q, k, v)
    ref = naive_attention(q, k, v)
    assert (out - ref).abs().max().item() < 5e-3


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA + triton")
def test_triton_non_causal():
    q = torch.randn(1, 2, 96, 64, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = triton_attention(q, k, v, causal=False)
    ref = naive_attention(q, k, v, causal=False)
    assert (out - ref).abs().max().item() < 2e-3
