# 《从零训练一个语言模型》教学书 — 章节索引

> 对标 xv6 book:每章 = 直觉 → 代码逐段(file:line)→ 动手实验 → 习题。
> 状态:☐ 未写 | 📝 写作中 | ✅ 完成(含已验证的可运行示例)
> 写作纪律:所有代码片段必须与本仓库 src/ 一致且可运行(pytest 通过);
> 每章末尾的习题答案放 book/answers/chNN.md,正文不放答案。

| 章 | 标题 | 对应代码 | 状态 |
|---|---|---|---|
| ch01 | 从字符预测说起:语言模型到底是什么 | 无(直觉章) | ✅ |
| ch02 | 分词器:文本如何变成数字 | src/tokenizer.py | ✅ |
| ch03 | 注意力:一个会"查字典"的函数 | src/model.py (Attention) | ✅ |
| ch04 | Transformer 块:RMSNorm、RoPE 与 SwiGLU | src/model.py (Block) | ✅ |
| ch05 | 训练循环:loss、AdamW 与学习率调度 | src/nn_utils.py, src/train.py | ✅ |
| ch06 | 让它变快:bf16、FlashAttention 与编译 | src/flash_attention.py, benchmark | ✅ |
| ch07 | 数据决定行为:两个语料,两个模型 | src/prepare_data.py, prepare_fineweb.py | ✅ |
| ch08 | 消融方法学:QK-Norm 实验复盘 | src/run_ablation.py | ✅ |
| ch09 | 部署:把模型装进 59MB | src/export_hf.py, thin-laptop-guide | ✅ |
| ch10 | 十个练习与下一步 | 全仓库 | ✅ |

## 写作规范(给每个写章节的 agent)

1. 开头 3 段讲直觉,不用公式;公式在"深入"小节才出现,且每个符号都解释
2. 引用代码必须带 `file:line`,且引用前实际读过该文件确认行号
3. 每章 ≥2 个"自己动手"(给出预期输出)+ ≥3 道习题(答案分离)
4. 章末"踩坑记录":引用 HANDOVER.md 对应条目
5. 中文写作,术语首现标英文
