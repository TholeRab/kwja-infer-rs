//! 流程编排（唯一同时依赖四个 crate 的地方）：
//! RAW -> normalize -> char tokenizer -> encoder.run_char -> char 三头 -> Juman++ 文本
//!     -> word tokenizer -> stride=2 分窗 -> encoder.run_word（逐窗）-> 12 头 -> 后处理 -> KNP 文本。

pub mod char_stage;
pub mod jumanpp;
pub mod knp;
pub mod window;
pub mod word_stage;
