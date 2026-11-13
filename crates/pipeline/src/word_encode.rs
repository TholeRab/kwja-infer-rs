//! word 阶段编码段：逐窗分词与编码器前向。

use kwja_tokenizer::word_side::WindowEncoding;

/// 编码预备的产物：分词结果 + 编码器输出。
pub(crate) struct WindowPrep {
    pub(crate) encoding: WindowEncoding,
    pub(crate) hidden: ndarray::Array3<f32>,
}
