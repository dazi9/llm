# GETTING_STARTED — 30 分钟从零到第一次生成

> 假设:你有一台 Windows 电脑(NVIDIA 显卡最好,没有也能跑)、会打开命令行。
> 结束时你将:在本地运行一个**你自己(参与)训练出来的语言模型**,并让它写一段文字。
> 遇到报错?直接翻本文件末尾的"踩坑对照表"——每一条都是真实发生过的事故。

## 第 0 步:克隆项目(1 分钟)

```powershell
git clone https://github.com/dazi9/llm.git
cd llm
```

## 第 1 步:进入 WSL Ubuntu(2 分钟)

项目在 WSL2 里训练和运行(Windows 原生也可以,但本指南走已验证的 WSL 路线)。

```powershell
wsl --status          # 没有?去 Microsoft Store 装 Ubuntu,然后回来
wsl -d Ubuntu
```

现在你在 Linux 里了。以下命令都在 Linux 中执行。

## 第 2 步:一键环境(10-20 分钟,取决于网速)

```bash
cd <你克隆的仓库路径>          # Windows 的 C:\Users\you\llm 在 WSL 里是 /mnt/c/Users/you/llm
bash scripts/env_setup.sh
```

这一步会:装 uv → 建 Python 3.13 虚拟环境 → 装 PyTorch(清华源)→
建立 `~/llm-lab/code` 符号链接(指向你的仓库)。

**预期输出**最后一行是 `=== env_setup done rc=0`。
看不到?看末尾踩坑表第 1、2 条。

## 第 3 步:拿一个模型(二选一)

**A. 拷现成的(如果你认识训练它的机器的主人)**:
把 `~/llm-lab/runs/v2/model_final.pt`(536MB)和 `~/llm-lab/data/tokenizer_web_32k.json`
从训练机拷到同样路径。

**B. 自己训一个小的(强烈推荐,~1 小时)**:

```bash
source ~/llm-lab/.venv/bin/activate
cd ~/llm-lab/code
bash scripts/download_data.sh          # 下载 TinyStories(~2GB)
python src/prepare_data.py --data-dir ~/llm-lab/data   # 训 BPE + 预分词(~5 分钟)
python src/train.py --preset small --compile --batch-size 64 \
    --max-steps 3000 --out ~/llm-lab/runs/my-first-model
```

看着日志里的 `loss` 从 9 一路掉到 1.6,你就亲眼见证了一次"语言模型的学习"。

## 第 4 步:第一次生成(1 分钟)

```bash
python src/sample.py --ckpt ~/llm-lab/runs/my-first-model/model_final.pt \
    --prompt "Once upon a time" --max-new 200
```

它会续写一个英文小故事。改 `--prompt` 试试别的开头;改 `--temperature 1.0` 看它发疯。

## 第 5 步:接下来去哪

- 想懂原理 → 读 `book/`(从 ch01 开始)
- 想动手 → 做 `labs/`(从 lab1 开始)
- 想部署到轻薄本 → 读 `reports/thin-laptop-guide.md`
- 想了解全局 → `HANDOVER.md`(运维手册)和 `reports/experiment-report.md`(实验报告)

## 踩坑对照表(每条都是真实事故)

| 你看到的报错 | 原因 | 解法 |
|---|---|---|
| `env_setup` 卡在 torch 下载 | pypi.nvidia.com 不可达 | 脚本已自动走清华源;确认没手动改回官方源 |
| `wsl.exe` 输出乱码 | 终端编码 | 无害,看最后的 exit code |
| `Cannot map tensor 'tok_emb.weight'` | 导出时没做张量名映射 | 用最新版 export_hf.py(已修复) |
| CUBLAS INTERNAL_ERROR 且重启无效 | 大概率是更早的静默断言(如 -100 索引) | 先修代码 bug,别信报错表面 |
| 训练吞吐突然掉 1/4 且功耗腰斩 | 显存顶格溢出到共享内存 | 减半 batch size |
| `pkill` 后自己的命令也没了 | pkill 匹配了自身命令行 | pkill 单独执行,模式加字符类 |
| push 显示 Connection reset | 网络 | 等几分钟重试;或走 SSH |
