//! 多标签头：word_feature（5 类）与 base_phrase_feature（64 类），
//! 把 sigmoid 概率判成 0/1。
//!
//! 概率由 `LoRASequenceMultiLabelingHead` 的
//! `torch.sigmoid(logits)` 输出；只做阈值解码，矩阵乘部分与其它头一样
//! 归推理接线。阈值语义取自两个消费点，必须逐条照抄：
//! - 基本句切分消费点：逐类 `>= 0.5`，
//!   外加基本句／文節切分的和阈值规则（`p[基本句-区切] + p[文節-区切] >= 1.0`）；
//! - BP 特征消费点：逐类 `>= 0.5`，
//!   其中 `節-区切`／`節-主辞` 两类由消费方特判，不走这里的逐类落键。

use crate::mask_tables::{BASE_PHRASE_FEATURES, WORD_FEATURES};

const _: () = {
    assert!(WORD_FEATURES.len() == 5, "WORD_FEATURES 应为 5 类");
    assert!(BASE_PHRASE_FEATURES.len() == 64, "BASE_PHRASE_FEATURES 应为 64 类");
};

/// WORD_FEATURES 里各类的下标（与 mask_tables::WORD_FEATURES 同序）。
pub const WORD_FEATURE_HEAD: usize = 0; // 基本句-主辞
pub const WORD_FEATURE_BP_END: usize = 1; // 基本句-区切
pub const WORD_FEATURE_PH_END: usize = 2; // 文節-区切
pub const WORD_FEATURE_YOGEN_HEAD: usize = 3; // 用言表記先頭
pub const WORD_FEATURE_YOGEN_TAIL: usize = 4; // 用言表記末尾

/// 逐类阈值判定：`p >= threshold` 时为 true。
///
/// 两个头的消费点都用 `>= 0.5`（含等号），用例里专门放了恰好 0.5 的行锁死这个语义。
#[must_use]
pub fn multi_label_flags(probs: &[f32], threshold: f32) -> Vec<bool> {
    probs.iter().map(|&p| p >= threshold).collect()
}

/// 基本句切分条件：
/// `p[基本句-区切] >= 0.5`，或两个切分概率之和 `>= 1.0`。
#[must_use]
pub fn closes_base_phrase(p_bp_end: f32, p_ph_end: f32) -> bool {
    p_bp_end >= 0.5 || p_bp_end + p_ph_end >= 1.0
}

/// 文节切分条件：
/// `p[文節-区切] >= 0.5` 且两个切分概率之和 `>= 1.0`。
#[must_use]
pub fn closes_phrase(p_bp_end: f32, p_ph_end: f32) -> bool {
    p_ph_end >= 0.5 && p_bp_end + p_ph_end >= 1.0
}
