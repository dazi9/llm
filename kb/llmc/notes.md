# llm.c 精读笔记

> 来源:`ref/llm.c/`(Karpathy,纯 C/CUDA 从零训练 GPT-2,~3000 行 C + CUDA kernel)
> 价值:看"没有 PyTorch 自动求导"时每一步长什么样 —— 所有前向/反向/优化器都是手写 kernel。

## 1. 仓库结构与管线

```
train_gpt2.py   # PyTorch 参考实现(先在 Python 里验证数学,再移植到 C)
train_gpt2.c    # CPU/单 GPU 版:C 手写前向+反向+AdamW
train_gpt2.cu   # CUDA 多 GPU 版:每层一个 kernel,NCCL all-reduce
llmc/*.cuh      # 按层拆分的 kernel:encoder/matmul/attention/layernorm/gelu/adamw/...
```

工作流是典型的"Python 原型 → C 移植":`test_gpt2.c` 先加载 PyTorch 导出的权重/激活
快照做逐张量对拍(与 CS336 的测试思想一致:用参考实现对拍手写实现)。

## 2. 内存计划(llmc 的核心贡献之一)

GPT-2 124M 前向有几十个中间张量。llm.c 在 `build_model` 里**一次性算出所有张量的
offset,分配一块连续显存池**,每层复用固定 slot:
- `num_activations = V*p + L*(...)`,README 示例:7330 万个激活 float(283MB 级);
- 反向时按"后进先出"的顺序取 slot,不经过 allocator。
对比 PyTorch:caching allocator 自动做类似的事,但 llm.c 的显式计划使显存可预算、
可在报告里精确解释 —— 这也是我们 benchmark 里 "peak VRAM" 列的意义。

## 3. 手写反向的串联顺序(train_gpt2.cu / llmc/*.cuh)

每层的反向严格按照依赖倒序:
```
dV = Pᵀ·dO → dP = dO·Vᵀ → dS = P∘(dP − D)/行归一 → dQ,dK → dln → dMLP → 残差累加
```
attention 反向用存下的 P(或 cudnn_att 用重算);`layernorm` 反向是方差重算 + 
`(x̂·d̂)的行和` 形式;`fused_classifier.cuh` 把 **softmax+CE+gradient 融成一个 kernel**
(logits 从不回 HBM)—— 这就是"为什么融合快"的最直白示范。

## 4. optimizer(adamw.cuh)

- `adamw_kernel3`:一个 kernel 处理整个参数向量:`m/v 更新 + bias correction + 
  decoupled wd + 参数写回`,还带 **master weights**(fp32 主权重,参数本身是 bf16);
- torch 的 fused AdamW 是同一个思路 —— 我们手写 AdamW 的 python 循环版在 100M 参数
  下每步多 ~几十个 kernel launch(见 nanogpt 笔记"改进清单")。
- `global_norm.cuh`:两遍法算全局梯度范数(reduce + scale),对应我们的
  gradient_clipping。

## 5. 多 GPU 与 ZeRO(zero.cuh)

- DDP:NCCL all-reduce 全部梯度(简单版);
- `zero.cuh`:ZeRO-1 —— 优化器状态(m/v/master)按 rank 分片,all-reduce 换成
  **reduce-scatter**(每 rank 只收自己分片)+ 更新后 all-gather 参数;通讯量不变但
  显存降为 1/num_ranks。CS336 A2 §5 同款。
- 多 GPU 逐 rank 数据加载:`dataloader.h` 按 rank 切分数据流,避免重复读。

## 6. 性能账本(README / 论文复现)

- GPT-2 124M 复现:8×H100 上 90 分钟训完 10B token,val loss ~3.28(原 GPT-2 论文 ~3.12
  是 8B tokens on WebText —— 可比性有限);
- `mfu.h` 提供 MFU 口径:flops_per_token = 6N + attention 项,除以 H100 bf16 峰值(989 TFLOPS);
- 单 GPU(CPU 模式)在 M3 Max 上一步(4×64)≈1.3s —— 教学演示足够,训练没戏。
  **结论:笔记本 CPU 训 LLM 不可行,但笔记本 GPU(5070 Ti,~40K tok/s)一夜可以推进
  10 亿 token 量级 —— 这正是我们项目的设定。**

## 7. 对本项目的借鉴(已吸收 / 待办)

| llm.c 做法 | 本项目 |
|---|---|
| Python 原型对拍 C 实现 | tests/ 对拍 torch(等价思想)✓ |
| fused softmax+CE | 我们 CE 用 logsumexp 两 kernel(compile 后自动融合一部分);专项融合列为改进 |
| 单 kernel AdamW + master weights | 手写 AdamW 循环版;fused 列为改进 |
| 显式显存计划 | torch allocator + peak VRAM 监控 ✓(报告口径) |
| dataloader 多进程 | prepare_data 20 进程预分词 + memmap ✓ |
| MFU 口径 | 报告补算 ✓ |
| outlier_detector.h(梯度离群监测) | 未做;如果 loss 毛刺可加 |

## 8. 补充精读:dataloader.h(02:55 追加)

llm.c 的数据加载器把"读文件"做成了一门小工程,几个值得记的点:

1. **.bin 文件 256 字节头约定**:header 里存 magic/version/数据维度,加载时校验,
   避免"拿错词表的数据训了半天"这类事故。我们的 memmap bin 没有头(靠 meta.json),
   属于简化;若要多实验共享数据,应升级为带头格式。
2. **分片(shard)流式读取**:glob 所有 shard → 按 shard 顺序逐个 fopen/fread,
   `current_shard_idx / current_sample_idx` 记进度;**uint16 缓冲 → int32 inputs/targets**
   在 fread 后一次转换。我们的 get_batch 用 numpy fancy indexing 做同样的事,但每次
   batch 现算;llm.c 预转换整块 —— 大 batch 时 IO 模式更规整。
3. **多进程偏移**:每个 rank 按 `process_rank / num_processes` 在 shard 内取互不重叠的
   样本段(`local_batch_offset_bytes`),天然支持 DDP 数据并行,不重复不遗漏。
4. **可选 shuffle**:mt19937 对 shard 顺序和 shard 内样本顺序做洗牌(多 epoch 时必须,
   否则模型记住 shard 顺序)。我们依赖 get_batch 的均匀随机采样,天然等效无序 ——
   这是 memmap 随机访问的额外好处(代价是随机 IO,TinyStories 4GB 全在 page cache,
   无实际影响)。
5. **错误处理宏**(utils.h):fopenCheck/freadCheck… 每个系统调用包裹检查 —— C 里没有
   异常,纪律靠宏。对比 Python:try/except 换成宏,本质相同。

## 9. 补充精读:nanoGPT sample.py 的工程细节(02:55 追加)

1. **torch.compile 模型加载 ckpt 的坑**:`state_dict` 里的 key 带 `_orig_mod.` 前缀
   (DDP/compile 包装产物),sample.py 显式剥离(unwanted_prefix 逻辑)。
   我们的 sample.py 直接保存未 compile 的 state_dict,天然没这个问题 —— 但如果将来
   存 `torch.compile(model).state_dict()`,要抄这段。
2. **采样也开 TF32**:`torch.backends.cuda.matmul.allow_tf32 = True` —— 推理路径同样受益。
3. **ctx 裁剪**:`idx[:, -block_size:]` 滑窗,我们 generate() 同款 ✓。
4. **meta.pkl 决定词表**:字符级数据集的 vocab 从 meta.pkl 恢复而不是硬编码 ——
   我们把 tokenizer 路径写进 ckpt,同思想但更直接。

## 10. 补充精读:train_gpt2.c 手写反向的完整调用序(03:25 追加)

train_gpt2.c:934-1004 的反向调用序是"手写 autograd"的教科书样板(逐行注释值得通读):

```
crossentropy_softmax_backward → matmul_backward(lm_head/wte) → layernorm_backward(lnf)
  每 block(倒序):
    residual_backward(dl_residual2, dl_fcproj, …)        # 残差分叉:梯度相加两路
    matmul_backward(fc_proj) → gelu_backward → matmul_backward(fc) → layernorm_backward(ln2)
    residual_backward(dresidual, dl_attproj, …)
    matmul_backward(attproj) → attention_backward(qkv) → matmul_backward(qkv) → layernorm_backward(ln1)
  encoder_backward(wte, wpe)
```

三个结构性观察:
1. **残差 backward = 梯度累加**:`residual_backward(dinp1, dinp2, dout)` 就是把 dout
   分别加进两路输入的梯度缓冲 —— PyTorch autograd 里 AddBackward0 干的事,这里是
   显式两行。看懂这个,autograd 就不再神秘。
2. **每个 `matmul_backward` 同时产出 dinp/dweight/dbias 三路梯度**(一次 Aᵀ·dout、
   dout·Bᵀ、行和),比 PyTorch 默认分开三个算子少两次中间量读写。
3. **attention_backward(dinp=dl_qkv) 是唯一需要前向激活回读的层**(P 或重算),
   对应 FlashAttention 论文里"存 logsumexp 重算 P"的动机(见 kb/cs336/a2 §2)。
   llm.c 的朴素版直接存了 (B,NH,T,T) 的 att 矩阵 —— 这正是它自己文档承认的
   显存大头,也是 FA 出现前的真实世界状态。

对照本项目:autograd 帮我们做了全部 1-3;我们能"写出来"的等价物只有 Triton FA forward
(fwd-only)。如果做 backward kernel,上面这条调用链就是蓝本。
