# llm-lab — 从零构建并优化小型语言模型(CS336 风格)

> 单卡 RTX 5070 Ti Laptop(12GB)+ WSL2 Ubuntu,全部组件手写:
> BPE / Transformer(RMSNorm+RoPE+SwiGLU+QK-Norm)/ AdamW / Triton FlashAttention / 消融 harness。
> 两条数据线:TinyStories(儿童故事,已完成)和 Fineweb-Edu(真实网页,最终模型)。

## 现在就能玩(不用等训练完)

进入环境(每次开终端先执行):
```bash
wsl.exe -d Ubuntu -e bash -lc "bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/env_setup.sh"   # 仅首次
source ~/llm-lab/.venv/bin/activate
cd /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab
```

**① 文本续写生成**(给开头,模型接 300 token):
```bash
# v1:故事模型(TinyStories)——写童话
python src/sample.py --ckpt ~/llm-lab/runs/main/model_final.pt --prompt "Once upon a time" --max-new 300 --n 3
# v2:网页语料模型(明早完成后可用)——写百科式文本
python src/sample.py --ckpt ~/llm-lab/runs/v2/model_final.pt --prompt "The history of artificial intelligence" --max-new 300 --n 3
# 参数:--temperature 0.8(越高越随机) --top-p 0.95 --seed 42(固定可复现)
```

**② 交互式续写 REPL**(打一句开头,回车,模型接着写;`exit` 退出):
```bash
python src/chat.py --ckpt ~/llm-lab/runs/v2/model_final.pt
```

**③ 微调后再玩**(SFT 让模型学会"给开头写完整段落并停止"):
```bash
python src/sft.py --init ~/llm-lab/runs/v2/model_final.pt --pairs ~/llm-lab/data/sft_pairs.jsonl --steps 300 --out ~/llm-lab/runs/sft_v2
python src/chat.py --ckpt ~/llm-lab/runs/sft_v2/model_final.pt
```

**⚠️ 期望管理(重要)**:这是 134M 参数的研究模型,不是 ChatGPT——
- 它做**续写**很好(给开头接得有模有样),但**不是指令聊天机器人**:你问问题它多半会"接着你的话写"而不是回答
- v2 只见过英文网页,中文输出会很差;英文流利但会一本正经地编造事实(所有 1 亿级模型的通病)
- 想要真·对话:把 SFT 数据换成指令数据集(如 Alpaca 子集,走 hf-mirror 可下)再跑一遍 `sft.py` 即可,管线已就绪

## 项目结构

```
llm-lab/
├── src/
│   ├── tokenizer.py        # BPE:GPT-2 正则 + 增量计数 + 惰性堆(262MB→10K 词表 20s)
│   ├── model.py            # RMSNorm/RoPE/SwiGLU/QK-Norm/GQA/NoPE 全部可开关
│   ├── nn_utils.py         # 手写 AdamW(与 torch 逐位对拍)/ cosine / 裁剪 / CE(-100 掩码)
│   ├── flash_attention.py  # Triton 手写 FlashAttention 前向(57 TFLOPS ≈ cuDNN 107%)
│   ├── train.py            # 训练循环:bf16/compile/断点续训/CSV 日志/消融开关
│   ├── run_ablation.py     # 受控架构消融驱动(参数自动对齐)
│   ├── aggregate_ablation.py
│   ├── benchmark.py / benchmark_adamw.py
│   ├── prepare_data.py     # TinyStories 管线;prepare_fineweb.py:流式 parquet 管线
│   ├── sample.py / chat.py / sft.py / build_sft_pairs.py
├── tests/                  # 23 个测试(AdamW 逐位对拍、kernel 正确性、续训回环…)
├── kb/                     # 精读笔记:CS336 A1/A2、nanoGPT、llm.c(+7 节追加)
├── reports/                # benchmark.md / ablation.md / experiment-report.md / 样本
├── runs-main/              # v1 训练日志归档(模型权重在 WSL ~/llm-lab/runs/)
└── scripts/                # env_setup / download_data / status / finalize
```

## 模型一览

| 模型 | 参数 | 数据 | token 量 | val loss | 位置(WSL) |
|---|---|---|---|---|---|
| v1 main | 100.3M | TinyStories | 9.05 亿 (1.94ep) | **1.2535** | runs/main/model_final.pt |
| 消融最优 small | 28.9M | TinyStories | 0.82 亿 | **1.5974** (qk_norm) | runs/ablation/qk_norm_lr1e-03/ |
| **v2 main** | **134.1M** | **Fineweb-Edu 网页** | **17.2 亿** | 训练中 | runs/v2/(ETA 次日 ~06:20)|

## 监控 / 收尾

```powershell
# 训练状态(Windows 任意终端)
wsl.exe -d Ubuntu -e bash -lc "tail -3 ~/llm-lab/runs/v2/train.log"
# 一键收尾(采样+SFT+归档,训练完成后)
wsl.exe -d Ubuntu -e bash -lc "bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/finalize.sh"
```

## 实验结果

- `reports/benchmark.md`:优化链 fp32+naive 42.9K → bf16+sdpa+compile **155.7K tok/s(3.63×)**;Triton FA 57 TFLOPS;AdamW 步进占比
- `reports/ablation.md`:QK-Norm 1.5974 / GQA 1.6206 / baseline 1.6294(排序跨学习率一致)
- `reports/experiment-report.md`:完整报告(含事故复盘与改进清单)
- `kb/`:CS336 A1/A2、nanoGPT、llm.c 精读笔记(中文)

## 网络环境说明(本机)

- huggingface.co 不可达 → hf-mirror.com;pypi.nvidia.com 不可达 → 清华源;github 可用但慢
