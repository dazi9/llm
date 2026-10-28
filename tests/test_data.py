import numpy as np
import torch

from data import TokenBin


def test_get_batch(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, size=5000).astype(np.uint16)
    p = tmp_path / "t.bin"
    p.write_bytes(tokens.tobytes())
    bin_ = TokenBin(str(p))
    gen = torch.Generator().manual_seed(0)
    x, y = bin_.get_batch(4, 8, gen, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # targets are inputs shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.dtype == torch.int64
    assert int(x.max()) < 1000
    # reproducible with same seed
    gen2 = torch.Generator().manual_seed(0)
    x2, y2 = bin_.get_batch(4, 8, gen2, device="cpu")
    assert torch.equal(x, x2)
