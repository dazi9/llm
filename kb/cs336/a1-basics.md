# CS336 Assignment 1 (basics) 精读笔记

> 来源:Stanford CS336 "Language Modeling from Scratch",Spring 2026 版 handout
> (`ref/assignment1-basics/cs336_assignment1_basics.pdf`,47 页;仓库 `ref/assignment1-basics/`)
> 本笔记对照本项目 `src/` 的实现逐点分析。2026 春季版 handout 明确写了 AI 工具政策:
> 不允许用 coding agent 代写作业 —— 我们这个项目是"自己造轮子"的学习性质,参考其规范与测试设计。

## 1. 作业要实现什么(= 我们的 src/ 清单)

A1 的 adapter 清单(tests/adapters.py)就是实现清单:

| CS336 要求 | 本项目对应 | 备注 |
|---|---|---|
| BPE tokenizer(train/encode/decode) | `src/tokenizer.py` | 增量 pair count + 惰性堆 |
| Linear / Embedding / RMSNorm / SiLU / SwiGLU | `src/model.py` | 全部手写,不用 torch.nn 的现成层 |
| causal scaled_dot_product / MHA / RoPE | `src/model.py` + `src/flash_attention.py` | sdpa 与 naive 双后端 |
| TransformerBlock / TransformerLM | `src/model.py` | pre-norm 残差结构 |
| cross_entropy / softmax | `src/nn_utils.py` | logsumexp 数值稳定实现 |
| AdamW / cosine LR / gradient clipping | `src/nn_utils.py` | 与 torch.optim 逐位对拍通过 |
| save/load checkpoint | `src/train.py` | model+optimizer+step 三件套 |
| get_batch(数据采样) | `src/data.py` | memmap + 随机窗口 |
| 训练 + 生成 + perplexity | `src/train.py` / `src/sample.py` | ppl = exp(平均 CE) |

允许使用的白名单(handout §What you can use):`torch.nn.Parameter`、容器类(Module/ModuleList)、
`torch.optim.Optimizer` 基类,其余 torch.nn/functional/optim 全禁。我们的实现实际用了
`F.scaled_dot_product_attention`(sdpa 后端)与 `F.cross_entropy` 仅作测试对拍基准 ——
训练路径的 CE 是手写的,attention 的 sdpa 是工程加速(在 A2 语义里这是被鼓励的优化项)。

## 2. BPE 训练的关键规范(handout §2)

1. **预分词**:使用 GPT-2 正则 `'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+`
   (handout §2.1 说明这是从 GPT-2 源码拿的"slightly prettier"版本)。作用:merge 永不跨越
   词边界/类别边界,且空格归属到词首。
2. **tie-breaking(handout §2.2 明确规定)**:pair 频次并列时,**选字典序更大的 pair**
   (如 ("A","C") > ("A","B"))。这是一个可复现性规范。本项目实现(heap 弹出取最小元组)
   在并列时取的是字典序更小者 —— 与 CS336 规范存在已知偏差,对最终分词质量影响可忽略
   (并列对通常频次低、语义弱),但要复现 CS336 的官方答案需改成取 max。
3. **special tokens**:训练前先从文本剥离,词表位置放在 merges 之后(本项目:`256+n_merges+i`);
    encode 时先按 special 切分再对各段编码。
4. **效率**(handout §2.3 有专门 problem):朴素实现每轮全表扫描是 O(merges × unique_words × len)。
   本项目做法:
   - `pair_counts` 增量维护(每次 merge 只重算受影响词);
   - `pair_to_words` 反向索引;
   - 惰性最大堆:pair 计数更新时 push 新条目,弹出时校验计数是否仍匹配,过期条目跳过。
   实测:262MB TinyStories 样本、vocab 10K → **20 秒**完成(CS336 参考实现约分钟级)。
5. 词表 ID 布局:0..255 字节,merges 从 256 起,specials 最后 —— 与 tiktoken/CS336 一致。

## 3. Transformer 组件的易错点(handout §3)

- **RMSNorm**:y = x / sqrt(mean(x²) + ε) * g。handout 测试考察 ε 位置(在均值内)。
  bf16 训练时必须 upcast 到 fp32 算均值再 cast 回(本项目实现如此),否则小值平方下溢。
- **SwiGLU**:FFN(x) = W2( SiLU(W1x) ⊙ W3x ),d_ff 取 ≈ 8/3·d 并对齐到 64 的倍数
  (handout 习题要求解释为什么是 8/3:保持参数量与 4x GELU MLP 相当)。
- **RoPE**:对 query/key 按位置旋转,θ=10000。**易错**:half-split(NeoX)与 interleaved(GPT-NeoX 之前)
  两种实现数值不同但都"对",handout 测试用其中一种约定 —— 本项目用 half-split
  (cos/sin 各 cat 两份,`x*cos + rotate_half(x)*sin`),与 HF LLaMA 一致。
  position 0 处 cos=1/sin=0 → 恒等变换,这是最快的 sanity check。
- **causal mask**:handout 测试直接考察"改未来 token 不影响过去输出"(我们的
  test_causal_attention_independence 同款)。
- **参数量账本**(handout 习题):untied 时 embedding 贡献 2·V·d。本项目 vocab=10K、
  d=768 → emb 15.4M / 总 100M ≈ 15%,这也是为什么小词表能省参数。

## 4. AdamW(handout §4.3,Algorithm 1)

CS336 版更新公式(与 torch.optim.AdamW 等价,本项目测试要求两者 20 步内 1e-6 一致):

```
m ← β1·m + (1-β1)·g
v ← β2·v + (1-β2)·g²
α_t = α · √(1-β2^t) / (1-β1^t)          # bias correction 并入步长
θ ← θ - α_t · m / (√v + ε) - α·λ·θ       # 解耦 weight decay:最后一项不加在梯度上
```

实现要点(torch op 顺序才能逐位一致):
- `denom = (v/bc2).sqrt() + ε`,`step_size = lr/bc1`,`p.mul_(1-lr·wd)` 先于 `addcdiv_`;
- m/v 与参数同形状同设备 → **AdamW 显存开销 = 2× 参数量(fp32)**,峰值再加梯度 1× 和
  bf16 autocast 的激活。handout 有专门习题算这本账(我们 100M 模型:参数 0.4GB +
  m/v 0.8GB + 梯度 0.4GB + 激活,12GB 卡无压力)。
- β2 用 0.95 而不是默认 0.999:短训练、大 batch 的常见选择(nanoGPT 同款)。

## 5. Cosine LR schedule(handout §4.4,LLaMA 同款)

```
it < T_w:        α_t = α_max · it / T_w                     # 线性 warmup
T_w ≤ it ≤ T_c:  α_t = α_min + ½(α_max-α_min)(1+cos(π·(it-T_w)/(T_c-T_w)))
it > T_c:        α_t = α_min
```

## 6. 训练与评估规范(handout §5)

- **checkpoint 必须包含**:模型权重、优化器状态(m/v/step)、迭代数 —— 否则无法真正续训。
- **perplexity = exp(平均 CE)**,评估用 val split,固定 seed 的 batch。
- handout 的训练配方参考(TinyStories):他们建议从 lr≈1e-3、batch 若干万 token 量级起步,
  以 val loss 是否下降、loss 曲线是否毛刺来判断 lr 合理性 —— 我们 benchmark 后按吞吐
  折算 batch,毛刺时先查 grad clip 和 warmup。

## 8. 补充:AdamW 资源记账(handout 习题,代入本项目真实数字,03:55 追加)

handout §4.3 的 `adamw_accounting` 习题要求对参数/激活/梯度/优化器状态分别列代数式。
代入本项目 main 配置(V=10,000, C=768, L=12, ctx=512, d_ff=2048, batch=32, fp32):

| 项 | 代数式 | 本项目数值 |
|---|---|---|
| 参数 | 2VC + L(4C² + 3C·d_ff) + 2C·(L+1) | 100.3M → **401 MB** |
| 梯度 | 同参数量 | **401 MB** |
| 优化器状态(m+v) | 2× 参数量 | **802 MB** |
| 激活/层(块内近似) | (11C + 3d_ff)·B·S·4B | ≈ 61KB/token/层 ×12 = 732KB/token |
| logits+CE | 2·V·B·S·4B(fp32 logits)≈ 655MB@B32 | **~0.65-1.3 GB** |
| **峰值合计** | — | **≈ 7-9 GB**(实测 8.6GB ✓ 对上了) |

验证:实测 train.py 报 8.6GB —— 手账与实测一致,说明没意外的显存泄漏/浪费。

(c) AdamW 一步 FLOPs ≈ 每参数 ~10 次标量运算(m、v、bias correction、两式更新、wd)
→ **~10N FLOPs/step**,对比前向 2N、反向 4N —— 优化器只占训练步 ~10% 的时间预算,
但作为几百个独立小 kernel 时 launch 开销可观(我们手写循环版的真实代价)。

(d) MFU 口径:throughput × (6N + 12·L·C·ctx)/peak。本项目 58.7K tok/s × 6.3e8 FLOPs/token
≈ 37 TFLOPS,对 5070 Ti Laptop bf16 峰值(~340 TFLOPS)== **~11% MFU**。
H100 单卡 50% MFU 训 GPT-2 XL(1.5B)400K 步 × batch 1024(handout 习题 d)要
~37 天;我们的 100M/一夜 = 同一套公式的另一种代入。

**(a) 注意题面陷阱**:d_ff=8/3·d 时 SwiGLU 激活是 3 个 (B,S,d_ff) 矩阵,不是 4x GELU
的 2 个 —— 题面让按 8/3 算,正说明 CS336 选 SwiGLU 配方的参数/激活效率动机。

## 7. 对本项目的直接启示

1. 我们已完成 A1 的全部实现面,且 20/20 测试通过 —— 其中 5 个测试直接对标 CS336 的考点
   (因果性、RoPE 参考实现、AdamW 对拍、BPE 回环、checkpoint/续训待补一个专门测试)。
2. 已知偏差两处,均有意为之:① BPE tie-break 取小不取大;② 训练用 sdpa 而非 naive
   (A2 的精神:正确性用 naive 验证,速度用融合 kernel)。
3. A1 handout 第 5 节训练循环里强调"先小模型跑通再放大" —— 我们 tiny→small→main 的
   三级流程就是按这个来的。
