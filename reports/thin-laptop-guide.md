# 轻薄本部署指南(把模型装进口袋)

> 目标:在无独显的轻薄本(Windows/Linux/macOS 均可,Intel/AMD/ARM 通吃)上运行
> 我们训练的模型。v1(100M,童话故事)已导出;v2 导出见文末注意。

## 第一步:拿到模型文件

已导出的 HF 格式在 WSL:`~/llm-lab/export/v1/`(config.json + pytorch_model.bin + tokens.json),
约 400MB。拷到轻薄本任意目录。

## 第二步:转 GGUF + 量化(在任一台有 Python 的机器上做一次)

```bash
git clone https://github.com/ggml-org/llama.cpp
pip install gguf  # 转换脚本依赖
python llama.cpp/convert_hf_to_llama.py <v1目录> --outfile model-f16.gguf
# 量化:536MB → ~130MB,质量损失可忽略
llama-quantize model-f16.gguf model-q4_0.gguf q4_0
```

## 第三步:在轻薄本上运行

下载 llama.cpp 官方预编译包(releases 页,选 CPU 版,AVX2 兼容 Intel/AMD/近五年所有机器):

```bash
llama-cli -m model-q4_0.gguf -p "Once upon a time" -n 200 --temp 0.8
# 交互模式:
llama-cli -m model-q4_0.gguf --interactive-first -c 512
```

预期速度:双通道内存的轻薄本约 60-120 token/秒;单通道约 30-60。
性能变量是内存带宽和线程数,与 CPU 厂商无关(`-t 8` 可手动指定线程)。

## 期望管理

- v1 是童话故事模型:给 "Once upon a time" 类开头,它会写出有对话、有结局的完整小故事
- 它只会续写,不会回答问题;事实会编造(134M 的物理极限,不是 bug)
- 中文输出很差(语料是英文)

## v2(134M 网页语料模型)注意

v2 开了 QK-Norm,LLaMA 格式的转换器**不认这个组件**,直接转换会得到错误的输出。
两个选择:① 等 llama.cpp 的 qk-norm 映射(或手改 convert 脚本,工作量不大);
② 重训一个不开 QK-Norm 的 v2 变体(QK-Norm 的收益 0.03 val,放弃不可惜)。
导出脚本 `src/export_hf.py` 会在 qk_norm 开启时打印警告。
