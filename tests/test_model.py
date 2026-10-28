import torch

from model import PRESETS, TransformerLM, _dff, build_model


def test_dff_rounding():
    assert _dff(256) == 704
    assert _dff(512) == 1344
    assert _dff(768) == 2048


def test_param_counts():
    for preset, d, L in [("tiny", 256, 4), ("small", 512, 6), ("main", 768, 12)]:
        m = build_model(preset, vocab_size=10000)
        n = m.num_params()
        emb = 2 * 10000 * d  # untied embedding + head
        per_layer = 4 * d * d + 3 * d * _dff(d)
        expect = emb + L * per_layer + 2 * d  # + 2 RMSNorm weights per block counted in per_layer? no
        # RMSNorm weights: 2 per block + 1 final, all d each -> included below
        expect = emb + L * (per_layer + 2 * d) + d
        assert abs(n - expect) < 0.001 * expect, (preset, n, expect)


def test_causal_attention_independence():
    torch.manual_seed(0)
    m = TransformerLM(vocab_size=64, d_model=64, n_layers=2, n_heads=4, ctx=16)
    m.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base = m(x[:, :4])
        full = m(x)[:, :4]
    assert torch.allclose(base, full, atol=1e-5), "future tokens leaked into past positions"


def test_rope_reference_match():
    from model import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 8)  # B, H, S, D
    out = rope(x)
    # reference: interleaved-pair rotation is NOT used; we use the NeoX half-split.
    d = 8
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(8).float()
    freqs = torch.outer(t, inv)
    cos = torch.cat([freqs.cos(), freqs.cos()], -1)
    sin = torch.cat([freqs.sin(), freqs.sin()], -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    ref = x * cos + torch.cat((-x2, x1), -1) * sin
    assert torch.allclose(out, ref, atol=1e-5)
    # norm preserved per position
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_forward_shapes_and_bf16():
    m = build_model("tiny", vocab_size=256)
    x = torch.randint(0, 256, (2, 32))
    logits = m(x)
    assert logits.shape == (2, 32, 256)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x)
    assert not torch.isnan(out.float()).any()


def test_residual_init_scale():
    import math
    L = 4
    m = TransformerLM(vocab_size=64, d_model=64, n_layers=L, n_heads=4, ctx=16,
                      residual_init_scale=True)
    expect = 0.02 / math.sqrt(2 * L)
    o_std = m.blocks[0].attn.o.weight.std().item()
    w2_std = m.blocks[1].mlp.w2.weight.std().item()
    assert abs(o_std - expect) / expect < 0.15, o_std
    assert abs(w2_std - expect) / expect < 0.15, w2_std
    # non-residual linears keep the plain 0.02
    q_std = m.blocks[0].attn.q.weight.std().item()
    assert abs(q_std - 0.02) / 0.02 < 0.15, q_std
    # default remains off
    m2 = TransformerLM(vocab_size=64, d_model=64, n_layers=L, n_heads=4, ctx=16)
    assert abs(m2.blocks[0].attn.o.weight.std().item() - 0.02) / 0.02 < 0.15
    # config round-trips for checkpoint save/load
    assert m.config["residual_init_scale"] is True
