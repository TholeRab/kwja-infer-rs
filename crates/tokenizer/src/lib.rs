//! 分词：char 侧与 word 侧各加载一份 tokenizer.json，产出与 Python 逐 id 相同的序列。

pub mod char_side;
pub mod word_side;

/// 一次编码的产物（input_ids / attention_mask 均为 ONNX 需要的 i64）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}
