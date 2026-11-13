//! 序列标注头与成对序列头（依存类型）。

use super::ops::{gelu, Linear};
use super::safetensors::WeightFile;

/// `SequenceLabelingHead`：Linear(hid, hid) + GELU + Linear(hid, labels)。
#[derive(Debug, Clone)]
pub struct SequenceHead {
    pub(in crate::forward) inner: Linear,
    pub(in crate::forward) outer: Linear,
}

impl SequenceHead {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        Self {
            inner: Linear::from_weights(w, &format!("{prefix}.0")),
            outer: Linear::from_weights(w, &format!("{prefix}.3")),
        }
    }

    /// 输入 (rows, hid)，输出 (rows, labels) 的 logits。
    ///
    /// 外层也必须走 `forward_rows`：reading 头的 labels 是 12907（读音词表），
    ///   逐行 `forward_row` 等于把「逐 token 的 matvec」原样搬到外层——
    ///   256 行 * 12907 标签的权重各扫一遍，单这一个头就要 1.26 s。
    pub fn logits(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mid = self.inner.forward_rows(hidden);
        let activated: Vec<Vec<f32>> = mid
            .iter()
            .map(|row| row.iter().map(|&v| gelu(v)).collect())
            .collect();
        self.outer.forward_rows(&activated)
    }
}


/// `SequenceLabelingHead(hidden*2, …)`（依存类型头）：
/// 输入是两个 hid 向量的拼接。
#[derive(Debug, Clone)]
pub struct PairSequenceHead {
    inner: Linear,
    outer: Linear,
}

impl PairSequenceHead {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        Self {
            inner: Linear::from_weights(w, &format!("{prefix}.0")),
            outer: Linear::from_weights(w, &format!("{prefix}.3")),
        }
    }

    /// argmax 标签（对最后一维取 argmax，并列取首）。
    pub fn predict_pair(&self, x: &[f32], y: &[f32]) -> usize {
        let mut concat = Vec::with_capacity(x.len() + y.len());
        concat.extend_from_slice(x);
        concat.extend_from_slice(y);
        let mid = self.inner.forward_row(&concat);
        let activated: Vec<f32> = mid.iter().map(|&v| gelu(v)).collect();
        let logits = self.outer.forward_row(&activated);
        crate::sequence_labeling::argmax(&logits)
    }
}

