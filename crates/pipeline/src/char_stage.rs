//! char 阶段（阶段 4 落地）：RAW 文本 -> 归一化 -> char 编码 -> 三头 argmax -> Juman++ 文本。
//! char 模型 batch 维固定为 1，只能逐窗跑（实测 batch=2 报 InvalidArgument）。
