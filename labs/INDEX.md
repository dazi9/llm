# Labs — 动手实验系列(本项目的 xv6 式灵魂)

> 每个 lab:README.md(目标/前置/步骤/验收标准)+ 骨架代码(TODO 标记)+
> answers/(参考答案,先不看你应该能做出来)。全部可用 `pytest` 验收。

| lab | 主题 | 骨架 | 预计耗时 | 状态 |
|---|---|---|---|---|
| lab1-bpe | 给 BPE 加字典序 tie-breaking,观察分词差异 | labs/lab1-bpe/ | 1-2h | ☐ |
| lab2-attention | 手写 naive attention 并与 sdpa 对拍 | labs/lab2-attention/ | 1-2h | ☐ |
| lab3-rope | 从二维旋转推导 RoPE,验证外推性质 | labs/lab3-rope/ | 1-2h | ☐ |
| lab4-adamw | 手写 AdamW 并与 torch 逐位对拍 | labs/lab4-adamw/ | 1-2h | ☐ |
| lab5-flash | 补全 Triton FlashAttention 的内层循环 | labs/lab5-flash/ | 2-3h | ☐ |
| lab6-ablation | 跑一次受控消融并写出报告 | labs/lab6-ablation/ | 3-4h | ☐ |
| lab7-deploy | 导出 GGUF 并在 CPU 上运行 | labs/lab7-deploy/ | 1h | ☐ |
| lab8-datamix | 数据配比实验:故事能力 vs 世界知识 | labs/lab8-datamix/ | 半天+GPU | ☐ |

## 验收纪律(给每个建 lab 的 agent)

1. 骨架代码必须**可运行但会失败**(TODO 处抛 NotImplementedError 或断言失败),
   学者补全后 pytest 转绿——这是 xv6 lab 的核心体验
2. answers/ 与骨架共用同一套测试,答案实现使测试通过
3. 每个 lab 独立可跑,不依赖其他 lab 的产出
4. README 里写明"你会卡在哪里"(基于 HANDOVER.md 的真实踩坑记录)
