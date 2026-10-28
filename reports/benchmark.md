# Throughput benchmark

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- model preset: small (build_model), batch 32 x seq 512, optimizer step included

| config | tokens/s | peak VRAM (GB) |
|---|---|---|
| fp32 + naive + eager | 42,881 | 7.62 |
| fp32 + sdpa + eager | 51,611 | 6.12 |
| bf16 + sdpa + eager | 109,559 | 4.89 |
| bf16 + naive + eager | 69,523 | 7.14 |
| bf16 + sdpa + compile | 155,656 | 3.93 |

## Attention forward microbench (B=8, H=12, S=512, D=64, bf16)

| impl | ms | TFLOPS |
|---|---|---|
| naive | 2.451 | 2.6 |
| sdpa(flash) | 0.120 | 53.6 |
| triton(fwd) | 0.113 | 57.1 |
| triton max|err| vs naive (fp32) | 8.76e-04 | - |

## Optimizer step microbenchmark (pure step, no fwd/bwd)

- model: small preset, 28,924,416 params (28.9M); scales ~linearly to 100M
- hand-written AdamW loops over ~200 tensors in Python; fused does it in one kernel

| implementation | ms/step |
|---|---|
| ours (hand-written, python loop) | 23.379 |
| torch AdamW (single-tensor) | 4.563 |
| torch AdamW (fused=True) | 5.944 |

- extrapolated to 100M params: ours ≈ 81.13 ms/step, fused ≈ 20.63 ms/step
- at 58.7K tok/s the training step itself is ~279 ms (16384 tokens/step), so the optimizer share is 29.1% (ours) vs 7.4% (fused) of step time
