"""End-to-end resume test: save checkpoint exactly like train.py does, load it
into a fresh model + fresh optimizer, and verify (a) weights round-trip,
(b) optimizer state (m/v/step) round-trip, (c) identical forward outputs.
Closes the gap noted in kb/cs336/a1-basics.md §7 (checkpoint/续训 test).
"""
import torch

from model import build_model
from nn_utils import AdamW


def test_checkpoint_resume_roundtrip(tmp_path):
    torch.manual_seed(0)
    m1 = build_model("tiny", vocab_size=256)
    opt1 = AdamW(m1.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    # take a couple of real optimizer steps so m/v are non-zero
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    for _ in range(2):
        opt1.zero_grad()
        logits = m1(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 256).float(), y.view(-1))
        loss.backward()
        opt1.step()

    # save exactly like train.py
    ckpt_path = tmp_path / "ckpt_last.pt"
    torch.save({"model": m1.state_dict(), "opt": opt1.state_dict(),
                "step": 2, "args": {"preset": "tiny", "vocab_size": 256}}, ckpt_path)

    # fresh process-equivalent: rebuild and load
    m2 = build_model("tiny", vocab_size=256)
    opt2 = AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m2.load_state_dict(sd["model"])
    opt2.load_state_dict(sd["opt"])

    # (a) weights identical
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), k1
    # (b) optimizer state identical (m, v, and step count)
    assert opt2.t == 2
    for a, b in zip(opt1.m, opt2.m):
        assert torch.equal(a, b)
    for a, b in zip(opt1.v, opt2.v):
        assert torch.equal(a, b)
    # (c) identical forward outputs after load
    m1.eval(), m2.eval()
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x), atol=0.0)

    # (d) continuing training from the resumed state reproduces the same update
    #     as continuing the original (identical grads, one step each)
    for (p1, p2) in zip(m1.parameters(), m2.parameters()):
        g = torch.randn_like(p1)
        p1.grad = g
        p2.grad = g.clone()
    opt1.lr = opt2.lr = 1e-4
    opt1.step(), opt2.step()
    for (p1, p2) in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)
