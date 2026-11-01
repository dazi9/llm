# HANDOVER — 接手指南(给下一个 agent 或人类工程师)

> 目标读者:从零接手本项目的 LLM 或工程师。读完这份 + README + reports/,你应该能
> 不问任何问题地继续运行和改进这个项目。本文件记录的是 README 里没有的"手感"。

## 1. 当前世界状态(以本文件 mtime 为准)

- **v2 训练**:134.1M(32K 词表 + QK-Norm),Fineweb-Edu 29 亿 token,batch 16 + compile,
  max_steps 120000(11.5 亿 token),以 `python -u` 挂在后台任务里,ckpt 每 5000 步,
  val 曲线见 runs/v2/log.csv。产出 `runs/v2/model_final.pt`。
- **已完成**:v1(100M TinyStories,val 1.2535)、消融 12 runs(QK-Norm 胜出 1.5974,
  reports/ablation.md)、SFT 管线验证、两套数据管线(TinyStories / Fineweb 流式)。
- **每 10 分钟有一个 cron 自检**(automation-1dc8136e):训练健康时它只做检查;
  注意它的 step-3/4/5 命令模板参数已过时(prepare 的 workers、v2 的 max_steps),
  以本文件为准,不要盲从 cron 里的命令。

## 2. 调试手册(本项目踩过的全部坑,按出现频率排序)

1. **pkill 自杀陷阱**:`pkill -f "xxx"` 会匹配自己所在复合命令里的任何 `xxx` 字面量
   → 杀死父 shell,静默无输出。规则:pkill 单独一个调用;模式用字符类破坏自匹配
   (`trai[n].py`);绝不与含目标串的重启命令写在同一条里。
2. **显存溢出的假象**:batch 32 + 32K 词表 → CE 的 FP32 logits 张量 ~2GB×数份 →
   11.78/12.2GB 触顶 → Windows WDDM **静默溢出到共享内存**,表现为 GPU 100% 但
   功耗 50W、吞吐掉到 1/4,无任何报错。诊断三件套:`nvidia-smi` 看 memory.used 是否
   顶格 + 功耗是否异常低 + py-spy 看是否长期停在 backward。药方:batch 减半。
3. **"GPU 100%" 是假象**:launch-bound / 共享内存爬行都会显示 100%。看功耗和
   实际吞吐,不看 util。
4. **mp.Pool.imap 急切 pickle 整个可迭代对象**:喂 70 万个文档字符串 = 主进程被
   pickle 队列撑爆 OOM。药方:窗口化(list(islice) + map 分块阻塞等待)。
5. **网页文本的分词缓存炸弹**:唯一词百万级,per-worker 缓存无界增长 → OOM。
   已在 tokenizer.py 加 100 万条上限(超限清空);worker 数 ≤6。
6. **巨型"无空格词"**(base64/压缩 JS):BPE 合并循环在 MB 级单词上爆炸。
   已加 >5000 字节回退字节编码;文档 >200KB 直接丢弃。
7. **WSL 服务会抽风**:高负载下 `Wsl/Service/WSAETIMEDOUT`,表现为所有 wsl 命令
   超时。等 90 秒重试,不行就 `wsl --shutdown`(训练会死,靠 ckpt + 自检恢复)。
8. **WSL 时钟漂移**:长会话后 WSL 时间可能落后 Windows 半小时以上。任何涉及
   "现在几点"的判断用 Windows 侧 `powershell Get-Date` 对表。
9. **Git Bash(MSYS)路径转换**:`/mnt/c/...` 或 `/change` 这类参数会被改写成
   `C:/Program Files/Git/...`。规则:wsl 命令一律 `bash -lc '整串'` 形式;
   Windows 原生命令(nvidia-smi/powercfg)加 `MSYS_NO_PATHCONV=1`。
10. **WSL 时区/时间与训练日志**:train.log 里的时间戳来自 WSL,和 Windows 墙钟
    可能差几十分钟,ETA 计算统一用吞吐倒推而不是日志时间戳。
11. **本机网络**:huggingface 不可达(hf-mirror.com 可用,偶发 15 字节错误页,
    下载必须做 Content-Length 校验);pypi.nvidia.com 不可达(torch 装清华源);
    github 慢但可用。
12. **交叉熵已支持 ignore_index=-100**(SFT 掩码)——早期版本 gather 越界会毒化
    CUDA 上下文,报出误导性的 CUBLAS 错误。若再遇 CUBLAS INTERNAL_ERROR,
    先怀疑上下文里更早的静默断言,不是 cuBLAS 本身。
13. **数据 bin 已 RAM 预载**(data.py preload=True):5.8GB 的 web bin 放进内存,
    memmap 随机读在它上面会磁盘抖动。若换更大的语料,注意 RAM 预算。

## 3. 怎么跑(标准操作)

- 训练:`python src/train.py --preset main --compile --batch-size 16 --max-steps N
  --train-bin ... --valid-bin ... --tokenizer ... --vocab-size 32000 --qk-norm`
  (32K 词表 + batch 16 = 显存 9.5GB,安全;batch 32 会溢出,别用)
- 状态:`wsl.exe -d Ubuntu -e bash -lc 'bash <项目>/scripts/status.sh'`(v1 用)
  或直接 tail runs/v2/log.csv
- 采样/REPL/SFT:见 README"现在就能玩"
- 断点续训:同命令 + `--resume`(cosine 会以当前 step 重新收尾到 max_steps,
  所以改小 max_steps 顺便就是"提前收尾"的正确姿势)
- 崩溃恢复顺序:看 py-spy(卡哪)→ dmesg(是否 OOM)→ nvidia-smi(显存顶格?)→
  按第 2 节对号入座

## 4. 实验方法学(为什么这么做)

- 消融必须:参数量对齐(变体间调 d_ff 补偿)、每变体扫 ≥3 LR、固定 token 预算、
  报告排序跨 LR 的一致性而不是单点差值
- 结论层级:单 seed 单尺寸的差值是噪声;排序一致才是信号;跨规模趋势才能写论文
- 参照系:GPT-2 124M 在 OWT 上 10B token → val ~3.1;我们 134M/7.3 亿 token
  Fineweb-Edu 预计 val ~2.85-2.95(语料更精)
- 所有报告落 reports/,所有教训落本文件——下一个 agent 请延续这个习惯

## 5. 还没做的事(按价值排序)

1. v2 训完后的:采样三连(科普/故事/列表开头)→ SFT → 报告 v2
2. 数据配比研究: TinyStories×Fineweb 混合比例 × 能力权衡(harness 现成)
3. Triton FlashAttention backward(公式在 kb/cs336/a2 §2)
4. 语音迁移:Interspeech 2027(截稿 2027-02-09),HuBERT 离散 token + 本 harness
5. fused AdamW 对比、残差缩放初始化已实现未消融

## 6. 环境事实卡

- GPU: RTX 5070 Ti Laptop 12GB(Blackwell sm_120,驱动 572.84);torch 2.9.1+cu128
  (pin 2.9.1:更新的版本解析 cu13 依赖,需驱动 ≥580,本机不行)
- WSL2 Ubuntu 26.04,15GB RAM(WSL 默认拿一半),ext4 955GB free
- 电源:已设"接通电源永不睡眠"(powercfg standby-timeout-ac 0)——**别改回去**,
  夜训就是被睡眠杀过一次才设的
- Python 3.13.18(uv 管理,venv 在 ~/llm-lab/.venv);系统 python 3.14 太新没轮子
