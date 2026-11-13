use std::collections::HashMap;

use candle::DType;
use serde::{Deserialize, Deserializer};

pub const DTYPE: DType = DType::F32;

// HiddenAct 与 HiddenActLayer 直接取自 candle-transformers 的 bert 实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

pub type Id2Label = HashMap<u32, String>;
pub type Label2Id = HashMap<String, u32>;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention: bool,
    pub max_relative_positions: isize,
    pub pad_token_id: Option<usize>,
    pub position_biased_input: bool,
    #[serde(deserialize_with = "deserialize_pos_att_type")]
    pub pos_att_type: Vec<String>,
    pub position_buckets: Option<isize>,
    pub share_att_key: Option<bool>,
    pub attention_head_size: Option<usize>,
    pub embedding_size: Option<usize>,
    pub norm_rel_ebd: Option<String>,
    pub conv_kernel_size: Option<usize>,
    pub conv_groups: Option<usize>,
    pub conv_act: Option<String>,
    pub id2label: Option<Id2Label>,
    pub label2id: Option<Label2Id>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<HiddenAct>,
    pub pooler_hidden_size: Option<usize>,
    pub cls_dropout: Option<f64>,
}

fn deserialize_pos_att_type<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize, Debug)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(s) => Ok(s.split('|').map(String::from).collect()),
        StringOrVec::Vec(v) => Ok(v),
    }
}
