//! word 模块 13 个头的前向：权重从 checkpoint 导出的
//! safetensors 读入（`assets/models/word_heads.safetensors`）。
//!
//! 算术与 torch 逐式对应：`Linear = x·W^T + b`、
//! `GELU = 0.5*x*(1+erf(x/sqrt(2)))`（torch `nn.GELU` 默认 `approximate='none'`）、
//! 多标签头最后 `sigmoid`。pairwise 头（dependency/cohesion/discourse）按
//! 「谓词行」惰性计算：每个 logit 元素的算术只依赖 (source i, target j) 这一对，
//! 只算解码要读的行，结果与整张算完逐元素相同；调用方还可只传目标行与候选列进一步缩减。
//!
//! safetensors 读入是手写的极简解析：8 字节小端头长 + JSON 头 + 数据区，
//! 只认 `F32`；不值得为这一个格式引依赖。权重整文件读入后按名拷出。

use std::path::Path;

use crate::mask_tables::{CONJFORM_TAGS, CONJTYPE_TAGS, POS_TAGS, SUBPOS_TAGS};
use safetensors::WeightFile;

mod lora;
mod ops;
mod relation_wise;
mod safetensors;
mod selection;
mod sequence;

pub use lora::LoraMultiLabelHead;
pub use ops::Linear;
pub use relation_wise::{LoraRelationWiseHead, RelationWiseProjection};
pub use selection::{SelectionHead, SelectionProjection};
pub use sequence::{PairSequenceHead, SequenceHead};

/// word 模块 13 个头 + CRF 的全部权重。
pub struct WordHeads {
    pub reading: SequenceHead,
    pub pos: SequenceHead,
    pub subpos: SequenceHead,
    pub conjtype: SequenceHead,
    pub conjform: SequenceHead,
    pub word_feature: LoraMultiLabelHead,
    pub ne: SequenceHead,
    pub base_phrase_feature: LoraMultiLabelHead,
    pub dependency: SelectionHead,
    pub dependency_type: PairSequenceHead,
    pub cohesion: LoraRelationWiseHead,
    pub discourse: SelectionHead,
    /// CRF 参数（Viterbi 解码用，解码函数在 `sequence_labeling`）。
    pub crf_start: Vec<f32>,
    pub crf_transitions: Vec<f32>, // (num_tags, num_tags) 行主序
    pub crf_end: Vec<f32>,
    pub num_ne_tags: usize,
}

impl WordHeads {
    pub fn load(path: &Path) -> Result<Self, String> {
        let w = WeightFile::load(path)?;
        let crf_start = w.f32("crf.start_transitions");
        let crf_transitions = w.f32("crf.transitions");
        let crf_end = w.f32("crf.end_transitions");
        let num_ne_tags = w.shape("crf.start_transitions")[0];
        Ok(Self {
            reading: SequenceHead::from_weights(&w, "reading_tagger"),
            pos: SequenceHead::from_weights(&w, "pos_tagger"),
            subpos: SequenceHead::from_weights(&w, "subpos_tagger"),
            conjtype: SequenceHead::from_weights(&w, "conjtype_tagger"),
            conjform: SequenceHead::from_weights(&w, "conjform_tagger"),
            word_feature: LoraMultiLabelHead::from_weights(&w, "word_feature_tagger"),
            ne: SequenceHead::from_weights(&w, "ne_tagger"),
            base_phrase_feature: LoraMultiLabelHead::from_weights(&w, "base_phrase_feature_tagger"),
            dependency: SelectionHead::from_weights(&w, "dependency_parser"),
            dependency_type: PairSequenceHead::from_weights(&w, "dependency_type_parser"),
            cohesion: LoraRelationWiseHead::from_weights(&w, "cohesion_analyzer"),
            discourse: SelectionHead::from_weights(&w, "discourse_relation_analyzer"),
            crf_start,
            crf_transitions,
            crf_end,
            num_ne_tags,
        })
    }

    /// 加载后做一次形状自检（标签数与常量表对齐；对不上说明权重与表不同源）。
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("pos", self.pos.outer.out, POS_TAGS.len()),
            ("subpos", self.subpos.outer.out, SUBPOS_TAGS.len()),
            ("conjtype", self.conjtype.outer.out, CONJTYPE_TAGS.len()),
            ("conjform", self.conjform.outer.out, CONJFORM_TAGS.len()),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!("{name} 头的标签数 {got} 与常量表 {want} 不一致"));
            }
        }
        if self.num_ne_tags != self.ne.outer.out {
            return Err(format!(
                "CRF 标签数 {} 与 NE 头 {} 不一致",
                self.num_ne_tags, self.ne.outer.out
            ));
        }
        Ok(())
    }
}
