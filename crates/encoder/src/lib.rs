//! encoder：推理后端抽象。当前后端为 ORT（`ort_backend`），以后可换 candle。
//! 本 crate 是全仓唯一为「换后端」留口子的地方，其余 crate 全部面向本 trait 编程。

use ndarray::Array3;

pub mod ort_backend;

/// 一次 word 前向的输入。字段语义与 ONNX 签名一一对应：
/// `word_encoder7.onnx` 输入 input_ids [batch,seq]、attention_mask [batch,seq]、
/// special_token_indices [batch,7]，全部 int64。
pub struct WordEncoderInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub special_token_indices: Vec<i64>,
    pub batch: usize,
    pub seq: usize,
}

/// 一次 char 前向的输入。`char_dynamic.onnx` 输入 input_ids [1,seq]、
/// attention_mask [1,seq]（batch 维固定为 1，seq 动态）。
pub struct CharEncoderInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub seq: usize,
}

/// char 阶段三个头的原始 logits。
pub struct CharLogits {
    /// (1, seq, 2)
    pub sent_segmentation: Array3<f32>,
    /// (1, seq, 2)
    pub word_segmentation: Array3<f32>,
    /// (1, seq, 6)
    pub word_norm_op: Array3<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("输入形状不符：{0}")]
    Shape(String),
    #[error("后端会话错误：{0}")]
    Session(String),
}

pub trait Encoder {
    fn run_char(&mut self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError>;
    fn run_word(&mut self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError>;
    fn backend_name(&self) -> &str;
    /// 启动时必须断言返回值含 CUDAExecutionProvider：
    /// ORT 曾静默回退 CPU，慢 18.8 倍且不报错（事实账 §六）。
    fn active_providers(&self) -> Vec<String>;
}
