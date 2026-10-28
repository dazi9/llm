# llm-lab — 从零构建并优化一个小型大语言模型(CS336 风格)

> One-night, from-scratch LLM build on a single RTX 5070 Ti Laptop (12GB), following
> the spirit of Stanford CS336 (Language Modeling from Scratch): hand-written BPE,
> hand-written Transformer, hand-written AdamW, hand-written Triton FlashAttention,
> measured optimizations. Everything runs inside WSL2 Ubuntu.

## 项目结构

```
llm-lab/
├── src/
│   ├── tokenizer.py        # 字节级 BPE:GPT-2 预分词正则 + 增量 pair count + 惰性堆训练;按 rank 合并的编码
│   ├── model.py            # RMSNorm / RoPE / SwiGLU / causal MHA(sdpa|naive 双后端),三个尺寸预设
│   ├── nn_utils.py         # 手写 AdamW(bias correction + 解耦 weight decay)、cosine 调度、梯度裁剪、交叉熵
│   ├── data.py             # uint16 token bin 的 memmap 随机批采样
│   ├── flash_attention.py  # Triton 手写 causal FlashAttention 前向(online softmax)+ naive 参考实现
│   ├── train.py            # 训练主循环:bf16 autocast / torch.compile / 断点续训 / CSV 日志
│   ├── benchmark.py        # 吞吐 benchmark:{fp32,bf16}×{naive,sdpa}×{eager,compile} + attention 微基准
│   ├── prepare_data.py     # BPE 训练 + 全语料多进程预分词 → memmap bin
│   ├── sample.py           # 加载 checkpoint,temperature/top-p 采样
│   ├── sft.py              # (prompt, completion) 对的 SFT,prompt 位置 mask 掉 loss
│   └── chat.py             # 故事续写 REPL
├── tests/                  # pytest:AdamW 对拍 torch、RoPE 参考实现、因果性、Triton 正确性、tokenizer 回环
├── scripts/                # env_setup.sh / download_data.sh(WSL 内执行)
├── kb/                     # 精读笔记(CS336 / llm.c / nanoGPT)
├── reports/                # benchmark 与实验报告
└── runs/                   # (WSL ~/llm-lab/runs) checkpoint 与训练日志
```

## 模型尺寸

| preset | d_model | layers | heads | d_ff(SwiGLU) | 参数量(vocab 10K, untied) |
|---|---|---|---|---|---|
| tiny   | 256 | 4  | 4  | 704  | ~8.3M  |
| small  | 512 | 6  | 8  | 1408 | ~29.6M |
| main   | 768 | 12 | 12 | 2048 | ~100M  |

统一 seq_len=512、RoPE θ=10000、RMSNorm ε=1e-5。数据:TinyStories(~2B tokens)。

## 快速开始(全部在 WSL 内)

```bash
# 0) 一次性环境(已由 agent 完成的话跳过)
bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/env_setup.sh

# 1) 数据准备(BPE 训练 + 全语料预分词,一次性,~10 分钟)
source ~/llm-lab/.venv/bin/activate
cd /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab
python src/prepare_data.py --data-dir ~/llm-lab/data

# 2) 冒烟(tiny,几分钟)
python src/train.py --preset tiny --max-steps 300 --out ~/llm-lab/runs/smoke

# 3) 正式训练(small ~1h;main 整夜)
python src/train.py --preset small --out ~/llm-lab/runs/small
python src/train.py --preset main --batch-size 32 --out ~/llm-lab/runs/main

# 4) 采样
python src/sample.py --ckpt ~/llm-lab/runs/main/model_final.pt --prompt "Once upon a time"

# 5) 基准(输出 reports/benchmark.md)
python src/benchmark.py --preset small

# 6) 跑测试
python -m pytest tests -q
```

## 监控训练(随时看)

```powershell
# Windows 侧一键状态(PowerShell / Git Bash 都行):
wsl.exe -d Ubuntu -e bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/status.sh
```

或直接在 WSL 里:`tail -f ~/llm-lab/runs/main/train.log`、`nvidia-smi`。
主力训练是 `setsid nohup` 脱离会话运行的,关掉终端不影响;**别关机/别睡眠**即可,
预计 ~07:25 完成 68000 步(≈11 亿 token ≈ 2.4 epoch),日志在 `runs/main/log.csv`。

## 实验结果(已回填部分)

- 优化链路:fp32+naive 42,881 tok/s → **bf16+sdpa+compile 155,656 tok/s(3.63×)**,详见 `reports/benchmark.md`
- 手写 Triton FlashAttention forward:**57.1 TFLOPS**,≈ cuDNN flash 的 107%,naive 的 24×
- small(28.9M)11 分钟训完,val loss **1.6414**,能生成完整有对话的故事(见 reports/)
- main(100.3M)整夜训练中;完整报告:`reports/experiment-report.md`
- 精读笔记:`kb/cs336/a1-basics.md`、`kb/cs336/a2-systems.md`、`kb/nanogpt/notes.md`、`kb/llmc/notes.md`

## 明早训练完成后可跑(可选)

**一键收尾**(采样 main → SFT 800 步 → 采样 SFT → 归档到工作区,约 5 分钟):
```powershell
wsl.exe -d Ubuntu -e bash -lc "bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/finalize.sh"
```

或分步(全部在 WSL 内):
```bash
source ~/llm-lab/.venv/bin/activate
cd /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab
python src/sample.py --ckpt ~/llm-lab/runs/main/model_final.pt --prompt "Once upon a time" --n 3
python src/chat.py --ckpt ~/llm-lab/runs/main/model_final.pt    # 交互式故事续写
# SFT(让模型学会"给开头写完整故事",~3 分钟):
python src/sft.py --init ~/llm-lab/runs/main/model_final.pt --pairs ~/llm-lab/data/sft_pairs.jsonl --steps 800
python src/chat.py --ckpt ~/llm-lab/runs/sft/model_final.pt
```

## 网络环境说明(本机)

- huggingface.co 不可达 → 数据走 hf-mirror.com
- pypi.nvidia.com 不可达 → torch 走 pypi.tuna.tsinghua.edu.cn(实测 ~7.4MB/s)
- github.com 可用但慢
