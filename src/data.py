"""Memory-mapped uint16 token bins + random batch sampling."""
from __future__ import annotations

import numpy as np
import torch


class TokenBin:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = len(self.data)

    def __len__(self) -> int:
        return self.n

    def get_batch(self, batch_size: int, seq_len: int, generator: torch.Generator,
                  device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        hi = self.n - seq_len - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        pos = ix.unsqueeze(1) + torch.arange(seq_len + 1)
        seq = torch.from_numpy(self.data[pos.numpy()].astype(np.int64))
        x = seq[:, :-1].to(device, non_blocking=True)
        y = seq[:, 1:].to(device, non_blocking=True)
        return x, y
