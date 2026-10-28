# nanoGPT 精读笔记

> 来源:`ref/nanoGPT/`(Karpathy,PyTorch 最小 GPT 训练器,~1000 行核心代码)
> 与本项目的相关性最高:同为 PyTorch + 单机 + 从零实现,工程细节几乎可以逐条对照。

## 1. model.py 逐模块

- **CausalSelfAttention**:`self.flash = hasattr(F, 'scaled_dot_product_attention')`(model.py:45)
  —— 双路径设计与我们一致:有 SDPA 就用 flash 路径,没有就用显式 mask 的 naive。
  注意它的 naive 路径用 `bias` 缓存因果 mask(tril),compile 时 mask 成为常量,有利融合。
- **MLP 是 4x GELU 不是 SwiGLU**(nanoGPT 忠实复刻 GPT-2);我们用 SwiGLU+RMSNorm+RoPE
  是 LLaMA 系配方 —— 两者都是合法选择,SwiGLU 参数效率更高,复刻 GPT-2 才用 GELU。
- **权重 tying**(model.py:135):`wte.weight = lm_head.weight`。我们选择 untied:词表只有
  10K,tying 省的 7.7M 参数(100M 的 7.7%)意义不大,untied 给输出层自由度。
- **残差投影缩放初始化**(model.py:142-144,来自 GPT-2 论文):
  `std = 0.02 / sqrt(2 * n_layer)` 专用于每个 Block 的最后一个投影(c_proj / 我们的 o、w2)。
  **这是本项目没有做的** —— 值得在后续实验加上(1 行改动),12 层深度的 100M 模型
  能得到更平稳的初始激活。(已列入"改进清单")
- **MFU 计算**(model.py:290-301):`flops_per_token = 6*N + 6*12*n_layer*d*seq`
  (前向 2N + 反向 4N,再加 attention 的 S 项),除以 GPU bf16 峰值。
  5070 Ti Laptop bf16 dense 峰值 ~340 TFLOPS(Blackwell,预热后),可以算我们的 MFU。

## 2. train.py 的工程细节

- dtype 选择(train.py:73):`bfloat16 if is_bf16_supported else float16`——fp16 需配
  GradScaler,bf16 不需要。我们直接 bf16(Blackwell 原生支持)✓
- **梯度累积边界处理**: iteration 内 `if (iter_num+1) % gradient_accumulation_steps != 0: continue`
  —— 只有累积满才 step/日志/eval,我们 train.py 的 accum 循环实现等价。
- grad clip 在 optimizer step 之前、所有 micro-batch backward 之后做(全局范数)✓ 与我们一致。
- **LR schedule**:warmup + cosine,`it > lr_decay_iters` 后保持 min(与我们同款)。
- 吞吐统计口径(train.py):**排除 eval/检查点时间**,取近 50 个 iteration 的滑动平均
  tokens/s。我们用 EMA(0.9 系数)+ 每窗口重置,口径一致。
- fused AdamW(train.py/model.py:281-283):`'fused' in signature(AdamW).parameters` 探测,
  CUDA 上必开。我们的手写 AdamW 是 python 循环版 —— 这是与 nanoGPT 的**最大性能差**:
  fused 版把 m/v 更新+参数更新合成一个 kernel。100M 参数下我们每步多付 ~几十个 kernel
  launch,约 2-5ms/步。(改进清单:用 torch._foreach_* 或直接换 fused,保留手写版做教学)
- torch.compile(train.py:204)直接 `torch.compile(model)` ✓ 同我们。

## 3. 参照系:README 里的复现数字(8×A100-40GB)

| 模型 | 训练时间 | loss | 备注 |
|---|---|---|---|
| GPT-2 124M | ~1 天 | ~3.11 | OpenWebText,~30B tokens |
| GPT-2 350M | ~1 天 | ~2.85 | |
| GPT-2 1.5B | ~1 个月 | ~2.5 | nanollama 系列用 10B tokens 量级 |

对照:我们的 main(100M,TinyStories 10K 词表)目标 val ~1.9-2.2(词表小 + 领域窄,
loss 天然低一个档;Loss 不可跨语料比较,只看趋势)。

## 4. 值得照抄的 10 个细节(本项目对照)

1. flash/naive 双路径留显式参考实现 ✓(我们已有)
2. bf16 + no GradScaler(条件允许时)✓
3. fused AdamW 探测 —— 待加
4. 残差投影 0.02/√(2L) 初始化 —— 待加
5. 排除 eval 的吞吐口径 ✓
6. checkpoint 存 model+optimizer+iter ✓
7. 采样用 top-k(我们用了 top-p,更好)+ 温度 ✓
8. `model.config` 随 ckpt 保存,加载时重建 ✓
9. MFU 上报口径 —— 报告里补
10. 一切实验先跑 tiny(玩具)验证管线再放大 ✓

## 6. 补充精读:resume 与 DDP 的工程细节(04:05 追加)

1. **resume 的配置合并策略**(train.py:157-176):只强制恢复影响张量形状的 6 个键
   (n_layer/n_head/n_embd/block_size/bias/vocab_size),其余(dropout 等)允许命令行
   改 —— "形状必须一致,超参可以变"是续训的黄金规则。我们的 resume 全量恢复 args,
   更保守;若要中途改超参(如调 max_steps 重设 cosine 终点),nanoGPT 的做法是对的。
2. **`_orig_mod.` 前缀剥离去重出现**(train.py:171-173):train.py 和 sample.py 各剥一次
   —— Karpathy 自己注释"honestly no idea how checkpoints sometimes get this prefix"。
   根因:compile 后的模型 state_dict 会带前缀,保存时应保存 unoptimized_model 的。
   我们在 train.py 里保存的是未 compile 的 model.state_dict() ✓ 免疫此坑。
3. **DDP 三件套**(train.py:82-101, 211-214):
   - `RANK/LOCAL_RANK/WORLD_SIZE` 环境变量判定,rank0 独占日志/存档;
   - **梯度累积除以 world_size**(`gradient_accumulation_steps //= ddp_world_size`),
     保证全局 batch 不随卡数变化;
   - DDP 包装后默认每 backward 全量 all-reduce;`model.no_sync()` 上下文用于累积的
     非最后 micro-batch(减少通讯)—— nanoGPT 没用 no_sync(每 micro-batch 都同步,
     简单但慢),llm.c 的逐参数 overlap 更进一步(CS336 A2 §5 要求实现的就是带
     no_sync + bucket 的版本)。
4. **seed 偏移**:`seed_offset = ddp_rank`,各卡数据不同;我们单卡用固定 seed ✓。
5. **estimate_loss 用 DDP 包装模型时**:DDP 的 forward 仍走 forward,eval 下
   no_sync 语义无影响;但 **estimate 内 model.eval() + no_grad** 与我们一致 ✓。
6. **crop_block_size 模型手术**(train.py:197-199):微调 GPT-2 时把 1024 ctx 裁到目标
   长度 —— 直接切权重矩阵的行。我们不需要(ctx 固定 512),但这是"改预训练模型形状"
   的标准手法。

## 5. nanoGPT → 我们的关键差异(为什么)

| | nanoGPT | 本项目 | 原因 |
|---|---|---|---|
| 位置编码 | 学习式 wpe | RoPE | 外推好、参数省;CS336 A1 用 RoPE |
| 归一化 | LayerNorm(带 bias) | RMSNorm(无 bias) | LLaMA 配方,省参数更稳 |
| MLP | 4x GELU | SwiGLU 8/3 | 同参数量下效果更好 |
| 词表 | 50257(GPT-2) | 10K(自训 BPE) | TinyStories 领域窄,小词表训练快 |
| 优化器 | torch fused AdamW | 手写 AdamW | 教学目的,fused 列入改进 |
