import torch
import torch.nn.functional as F

from nn_utils import AdamW, cosine_lr_schedule, cross_entropy, gradient_clipping


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(64, 101) * 5
    targets = torch.randint(0, 101, (64,))
    ours = cross_entropy(logits, targets)
    ref = F.cross_entropy(logits, targets)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_adamw_matches_torch():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(30, 20))
    p2 = torch.nn.Parameter(torch.randn(10))
    ref1, ref2 = torch.nn.Parameter(p1.detach().clone()), torch.nn.Parameter(p2.detach().clone())
    ours = AdamW([p1, p2], lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    ref = torch.optim.AdamW([ref1, ref2], lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    for i in range(20):
        g1 = torch.randn_like(p1) * (i + 1)
        g2 = torch.randn_like(p2) * (i + 1)
        p1.grad = g1.clone()
        p2.grad = g2.clone()
        ref1.grad = g1.clone()
        ref2.grad = g2.clone()
        ours.step()
        ref.step()
    assert torch.allclose(p1, ref1, atol=1e-6), (p1 - ref1).abs().max()
    assert torch.allclose(p2, ref2, atol=1e-6)


def test_cosine_schedule_values():
    lr = cosine_lr_schedule
    assert lr(0, 1.0, 0.1, 100, 1000) == 0.0
    assert abs(lr(100, 1.0, 0.1, 100, 1000) - 1.0) < 1e-9
    assert abs(lr(1000, 1.0, 0.1, 100, 1000) - 0.1) < 1e-6
    mid = lr(550, 1.0, 0.1, 100, 1000)
    assert 0.1 < mid < 1.0
    # monotone decreasing after warmup
    assert lr(400, 1.0, 0.1, 100, 1000) > lr(700, 1.0, 0.1, 100, 1000)


def test_gradient_clipping():
    p = torch.nn.Parameter(torch.zeros(10))
    p.grad = torch.ones(10) * 10  # norm = sqrt(10)*10 = 31.6 > 1 -> clipped to 1
    gradient_clipping([p], max_norm=1.0)
    assert abs(torch.linalg.norm(p.grad) - 1.0) < 1e-5
    # below threshold -> untouched (norm = 0.03*sqrt(10) = 0.095 < 1)
    p.grad = torch.ones(10) * 0.03
    before = torch.linalg.norm(p.grad).item()
    gradient_clipping([p], max_norm=1.0)
    assert abs(torch.linalg.norm(p.grad).item() - before) < 1e-7
