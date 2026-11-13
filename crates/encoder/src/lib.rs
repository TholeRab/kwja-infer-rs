//! encoder：推理后端抽象。其余 crate 一律面向 Encoder / ParallelEncoder trait 编程，
//! 不直接依赖具体后端实现（ort_backend、candle_backend）。

use std::path::Path;

use ndarray::Array3;

pub mod candle_backend;
pub mod debertav2;
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

/// char 侧三个序列标注头的原始 logits。
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

/// 按 KWJA_BACKEND 选后端：缺省 ort（ONNX Runtime，DLL）；
/// candle/rust = 纯 Rust candle 后端（无 ONNX Runtime、无 DLL），权重为 ckpt
/// 转出的 safetensors（tools/convert_ckpt_to_safetensors.py），路径可用
/// KWJA_WORD_CKPT / KWJA_CHAR_CKPT 覆盖。
/// device 只对 ort 有意义（candle 恒为 CPU，多线程靠统一 rayon 池）。
pub fn make_backend(
    char_model: impl AsRef<Path>,
    word_model: impl AsRef<Path>,
    device: ort_backend::Device,
) -> Result<Box<dyn Encoder>, EncoderError> {
    match std::env::var("KWJA_BACKEND").unwrap_or_default().as_str() {
        "candle" | "rust" => Ok(Box::new(candle_backend::CandleBackend::new(
            char_model, word_model,
        )?)),
        _ => Ok(Box::new(ort_backend::OrtBackend::new(
            char_model, word_model, device,
        )?)),
    }
}

/// 能被多线程共享调用的后端（&self、Sync）。ORT 的 session 是 &mut 独占，
/// 且其 CPU EP 自带线程池，与 rayon 抢同一批核；
/// tract/candle 的前向不可变，编码可与解码/合并共用同一个 rayon 池。
pub trait ParallelEncoder: Encoder + Sync {
    fn run_word_shared(&self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError>;
    fn run_char_shared(&self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError>;
}

/// Send：编码需能挪到独立线程独占运行（见 pipeline word_predict 的双缓冲）。
pub trait Encoder: Send {
    fn run_char(&mut self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError>;
    fn run_word(&mut self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError>;
    /// 若后端支持共享调用（&self、Sync），返回它的并行视图；ORT 返回 None。
    fn as_parallel(&self) -> Option<&dyn ParallelEncoder> {
        None
    }
    fn backend_name(&self) -> &str;
    /// 启动时必须断言返回值含 CUDAExecutionProvider：
    /// ORT 曾静默回退 CPU，慢 18.8 倍且不报错。
    fn active_providers(&self) -> Vec<String>;
}
