//! 流程编排（唯一同时依赖四个 crate 的地方）：
//! RAW 经 normalize、char tokenizer、encoder.run_char 与 char 三头得到 Juman++ 文本；
//! 再经 word tokenizer、stride=2 分窗、encoder.run_word（逐窗）、12 头与后处理得到 KNP 文本。

pub mod char_stage;
pub mod char_tables;
pub mod char_tags;
pub mod chunking;
pub mod cohesion;
pub mod dependency;
pub mod discourse;
pub mod jinf;
pub mod jumandic;
pub mod jumanpp;
pub mod knp;
pub mod knp_emit;
pub mod morpheme_build;
pub mod norm_op;
pub mod window;
mod word_decode;
mod word_encode;
mod word_merge;

pub mod word_predict;
pub mod word_stage;
pub mod word_tables;
