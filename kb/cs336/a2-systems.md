# CS336 Assignment 2 (systems) 精读笔记

> 来源:`ref/assignment2-systems/cs336_assignment2_systems.pdf`(48 页,Spring 2026)
> 主题:把 A1 的"能训"变成"训得快"。本项目 benchmark 的方法论与 kernel 工作直接对标此篇。

## 1. A2 的六个部分(handout 目录)

1. operator fusion / 算子融合基线
2. benchmark 神经网络操作的方法论(含 torch.compile 的归因问题)
3. **FlashAttention-2 Triton kernel**(forward + backward)
4. 分布式数据并行(DDP)
5. 优化器状态分片(ZeRO-1)
6. Fully sharded data parallel(FSDP)

## 2. 为什么要融合:attention 的 IO 分析(handout §4.1.1)

vanilla attention 的显存与 IO 瓶颈不在计算而在搬运:
- P = softmax(QKᵀ/√d) 是 (B, H, S, S) 的巨型中间张量,反向要两次从 HBM 读它
  (算 dV 用一次、算 dS 用一次);
- 显存占用随 S² 增长 → 长 seq 直接 OOM。

FlashAttention 的三板斧:
1. **Tiling**:把 Q/K/V 分块,softmax 用 online 方式增量归约(never materialize 全矩阵);
2. **Recomputation**:前向只存 logsumexp L,反向重算 P(用 exp(S−L) 一条指令);
3. **Fusion**:整个前向一个 kernel,HBM↔SRAM 搬运量从 O(S²) 降到 O(S)。

**反向公式(FA2 论文版,存 L 后无需 softmax)**:先预计算 D = rowsum(O∘dO),
然后 dS_ij = P_ij(dP_ij − D_i),其余 dV = PᵀdO、dQ = dS·K/√d、dK = dSᵀ·Q/√d。
本项目只手写了 forward(见下),训练路径用 SDPA 的 cuDNN flash 后端(其反向是
NVIDIA 的 FA 实现)——这是工程上正确的取舍。

## 3. 本项目 Triton kernel 的实现要点与实测(src/flash_attention.py)

- 网格:grid = (ceil(S/BLOCK_M), B·H),每个 program 算一个 query tile;
- 内层循环 K/V tiles:running max m、running sum l、acc 三件套在线 softmax;
- **数值技巧**:全部用 exp2 实现 exp(p = exp2((s−m)·log₂e)),Tensor core 时代 exp2 更快;
  完全被 mask 的行 m=-inf 时用 m_safe 防止 NaN;
- **TF32 坑**:tl.dot 对 fp32 输入默认走 TF32(10 位尾数),和 naive fp32 参考比对会有
  ~3e-3 误差 —— 不是 bug,要么放宽容差要么 `input_precision="ieee"`;
- 常量必须是 `tl.constexpr`(Triton 3.5 起 kernel 内不能引用普通模块级变量);
- **Windows/WSL 依赖**:Triton 需要 gcc + libc6-dev 编译 driver(Ubuntu 26.04 默认没有)。

实测(RTX 5070 Ti Laptop,B=8,H=12,S=512,D=64,bf16):

| 实现 | 耗时 | TFLOPS |
|---|---|---|
| naive(显式 mask) | 2.45 ms | 2.6 |
| F.scaled_dot_product_attention(cuDNN flash) | 0.120 ms | 53.6 |
| **本项目 Triton FA fwd** | **0.113 ms** | **57.1** |

手写 kernel 达到 cuDNN 的 ~107%(本次跑分甚至略快),naive 的 ~24 倍。
(注:SDPA 和 Triton 各次运行波动 ±10%,结论是"同档位",不是"稳定更快"。)

## 4. 整模型吞吐 benchmark 的方法论(handout §2 + 本项目实现)

- 必须包含 warmup(首步含 kernel 编译/autotune/clock ramp,我们的冷启动 matmul 只有
  9 TFLOPS,稳态后几十倍于此);
- torch.compile 的归因问题(handout 明确警告):compile 会把很多 op 融成一个 kernel,
  无法再按层归因 —— 归因分析要在 eager 下做;
- 报告要同时给 tokens/s 与 peak VRAM;
- 本项目结果(small preset,batch 32×512,含优化器 step):

| 配置 | tokens/s | 相对 fp32+naive | peak VRAM |
|---|---|---|---|
| fp32 + naive + eager | 42,881 | 1.0× | 7.62 GB |
| fp32 + sdpa | 51,611 | 1.20× | 6.12 GB |
| bf16 + naive | 69,523 | 1.62× | 7.14 GB |
| bf16 + sdpa | 109,559 | 2.56× | 4.89 GB |
| **bf16 + sdpa + compile** | **155,656** | **3.63×** | **3.93 GB** |

结论:精度(bf16)与融合(sdpa/compile)是**乘法关系**;compile 一项再 +42%;
显存反而更省(flash attention 消掉 S² 激活 + compile 融掉中间量)。

## 5. 分布式训练要点(handout §5-6,本项目单卡,记录备用)

- backend:GPU 用 NCCL,CPU 才用 Gloo;
- 朴素 DDP:backward 结束后一次 all-reduce 全部梯度 → 通讯不与计算重叠;
- **DDP + overlap**:按参数就绪即发 all-reduce(handout 的 ddp_overlap_individual_parameters),
  bucket 化:小梯度先攒 bucket 再通讯,减少小包;5 轮 warmup 后再计时;
- ZeRO-1:优化器状态(m/v)按 rank 分片,每 rank 只存自己负责参数的 m/v,
  all-reduce 改 reduce-scatter + 参数 all-gather,显存省 ~num_ranks 倍;
- FSDP 再进一步把参数也分片,前向/反向时临时 all-gather,用完即弃。
- 单卡 12GB 的本项目:DDP/ZCP 用不上,但 benchmark 表就是 A2 §"DDP benchmarking" 要的
  同款表格;若未来上多卡,train.py 的优化器/调度/日志结构可直接复用。

## 7. 补充精读:分布式通讯量的环分析法(04:25 追加,handout §8)

A2 后半部分把"选哪种并行"变成了一道通讯量计算题,方法比结论更值得记:

1. **通讯原语的环实现与代价**(N 卡、每卡出口带宽 W、张量 S 字节):
   - ring all-gather:N-1 步,每步每人发 S/N → **时间 (N-1)/N · S/W**;
   - ring reduce-scatter:同结构、传的换成分块部分和 → **同为 (N-1)/N · S/W**;
   - ring all-reduce = reduce-scatter + all-gather → **2(N-1)/N · S/W**;
   - 关键直觉:all-reduce 的有效带宽是点对点的 **2(N-1)/N ≈ 2×**(N 大时),因为
     每卡同时收发、利用了全拓扑,而不是一个中心 server。
2. **各策略的每步通讯量**(模型 N_p 参数、激活 A、batch B):
   - DP/DDP:all-reduce 梯度 ≈ 2(N-1)/N · N_p;
   - ZeRO-1/FSDP(仅优化器分片):reduce-scatter 梯度 + all-gather 参数
     = (N-1)/N · N_p + (N-1)/N · N_p ≈ **与 DDP 同通讯量但显存省 N 倍**(m/v/参数分片);
   - FSDP(全分片):再加前向/反向的参数 all-gather ×2 → 通讯量 ~2× DDP,显存 ~1/N;
   - TP:每层前向+反向都要通讯(量 ∝ B·S·d),**对带宽最敏感**,一般只放机内 NVLink;
   - PP:只传层边界激活(量 ∝ B·S·d,次数少)但配比失衡时有 bubble。
3. **选型法则**:通讯时间 = 通讯量/带宽,计算时间 = FLOPs/算力;两者之比告诉你
   在多少卡之前通讯不成为瓶颈。组合策略(如 DP×TP)就是把两种通讯串行叠加后重新
   算这个比值。
4. **A2 的 DDP 实现阶梯**(§5):朴素 all-reduce → 分桶(bucket)减少小包 →
   逐参数异步 all-reduce + backward 重叠(async handle + finish_gradient_synchronization)
   → 用 Nsight 时间线图证明重叠真的发生了 —— "声称重叠"和"证明重叠"是两件事,
   这是 A2 最工程化的地方。

对照本项目:单卡用不上这些,但 benchmark 表的"通讯墙"思想一致 —— 12GB 显存限制下
我们做的是**批大小 × 序列长度的显存-吞吐权衡**,和分布式做的是同一类预算分配题。

## 8. 补充精读:profiling 方法论(04:55 追加,handout §2.1.4-2.1.5)

A2 用 5 分的大题逼学生做 Nsight Systems 剖析,考点值得记:

1. **nsys 命令行包**:`nsys profile --trace=cuda,osrt,nvtx --pytorch --cudabacktrace=all
   --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py`。
   各开关都有代价:`--cudabacktrace=all`/`--python-backtrace=cuda` 开销特别大,
   不需要回溯时应去掉 —— **profiling 本身会拖慢运行**,按需开启。
2. **NVTX range 标注**:用 `torch.cuda.nvtx.range("...")` 给代码段(如 attention 内部的
   scores/softmax/ matmul 三段)打标签,profile 里自动出现分段块;再配合
   `--nvtx-capture` 过滤掉 warm-up 步 —— **排除 warm-up 是基准测试纪律**(我们的
   benchmark.py 用 warmup 轮次 + 计时窗口实现同一目标)。
3. **monkey-patch 剖析法**:把被测函数替换成带标注版本
   (`cs336_basics.model.scaled_dot_product_attention = annotated_...`),不用改库代码。
4. **torch.compile 的归因警告**(handout 原话:can make it hard to attribute time and
   resources):compile 把多个 op 融成一个 kernel 后,NVTX 标签和 kernel 名都对不上源码
   —— 归因分析必须在 eager 下做,compile 只用于端到端吞吐对比。我们的流程正是:
   eager 归因(microbench)→ compile 端到端(吞吐表)。
5. **nsys 习题的五连问**(都是好问题):前向最耗时的 kernel 是哪个、调几次?
   forward+backward 后最大 kernel 会不会换人?非 matmul kernel 占了多少(答案通常是
   elementwise/归约类)——这正是"为什么融合重要"的实证;AdamW step 在整步训练里
   占比多少(手写多 kernel 版会更显眼);softmax 与 matmul 的耗时比 vs FLOPs 比
   (memory-bound 与 compute-bound 的分野,也是 FlashAttention 的立论)。

对照本项目:我们没上 nsys(Windows/WSL 下要装 CUDA toolkit + nsys,今晚不值当),
但 attention 微基准(0.113ms vs 2.45ms)已经回答了第 (e) 问:naive attention 是
memory-bound 的,所以"减少 HBM 读写"的融合能把 24 倍时间省掉。

## 6. 对本项目的启示清单

1. 我们已完成 A2 的 §1-3(融合基线、benchmark 方法论、Triton FA fwd),§5-6 留作多卡扩展;
2. A2 强调 backward kernel 同样要写 —— 我们的 FA 只写了 fwd,反向依赖 SDPA;
   若要补齐:按 §4.1.1 的公式(13)-(19) + 存 L,是下一个自然的 extension;
3. compile 的正确姿势:先 eager 冒烟确认数值,再开 compile(compile 会掩盖数值 bug);
4. 所有速度结论必须配"同 batch/seq/精度"的对照表 —— 这正是我们 benchmark.md 的格式。
