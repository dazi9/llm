# Architecture ablation — small preset, 2500 steps x 32K tokens (82M tokens/run),
# batch 64x512, bf16+sdpa+compile, param-aligned (gqa2 via d_ff=1600), seed 0

| variant | lr | val@1000 | val@2000 | val@2500 (final) | tok/s |
|---|---|---|---|---|---|
| qk_norm | 1e-03 | 1.8466 | 1.6347 | **1.5974** | 152,503 |
| gqa2 | 1e-03 | 1.8835 | 1.6621 | **1.6206** | 158,553 |
| baseline | 1e-03 | 1.897 | 1.6695 | **1.6294** | 161,439 |
| qk_norm | 6e-04 | 1.9098 | 1.6947 | **1.6595** | 142,987 |
| gqa2 | 6e-04 | 1.9255 | 1.7137 | **1.6783** | 158,159 |
| baseline | 6e-04 | 1.9492 | 1.7227 | **1.6847** | 156,286 |
| gqa2 | 3e-04 | 2.0617 | 1.8346 | **1.8015** | 156,719 |
| qk_norm | 3e-04 | 2.0832 | 1.8451 | **1.8113** | 134,964 |
| baseline | 3e-04 | 2.0931 | 1.8501 | **1.8158** | 157,222 |

## Best per variant (min final val)

- **qk_norm**: 1.5974 @ lr 1e-03
- **gqa2**: 1.6206 @ lr 1e-03
- **baseline**: 1.6294 @ lr 1e-03
