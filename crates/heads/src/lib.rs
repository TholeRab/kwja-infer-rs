//! 13 个头：全部是矩阵乘 + argmax（或多标签阈值），不依赖任何推理框架，与后端完全解耦。

pub mod forward;
pub mod lora_multi_labeling;
pub mod lora_relation_wise;
pub mod mask_tables;
pub mod sequence_labeling;
pub mod word_selection;
