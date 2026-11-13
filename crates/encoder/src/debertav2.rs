//! 基于 candle-transformers 0.9.2 的 debertav2.rs 改造：加入 KWJA 特有的
//! special_token_indices 相对位置扩展——special token 参与的 (query, key) 组，
//! 相对位置改用 -position_buckets，对角线保持 0。
//! 结构与 transformers 的 DeBERTa-v2 参考实现逐块对应。

mod attention;
mod config;
mod conv;
mod embeddings;
mod encoder;
mod feed_forward;
mod layer;
mod layers;
mod model;
mod relpos;
mod tasks;

pub use attention::{DebertaV2Attention, DebertaV2DisentangledSelfAttention};
pub use config::{Config, DTYPE, HiddenAct, Id2Label, Label2Id};
pub use conv::ConvLayer;
pub use embeddings::DebertaV2Embeddings;
pub use encoder::DebertaV2Encoder;
pub use feed_forward::{DebertaV2Intermediate, DebertaV2Output, DebertaV2SelfOutput};
pub use layer::DebertaV2Layer;
pub use layers::{HiddenActLayer, StableDropout};
pub use model::DebertaV2Model;
pub use tasks::{
    DebertaV2ContextPooler, DebertaV2NERModel, DebertaV2SeqClassificationModel, NERItem,
    TextClassificationItem,
};

pub(crate) use relpos::{build_relative_position, make_log_bucket_position};
