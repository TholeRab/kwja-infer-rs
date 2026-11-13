//! LoRA 增量与多标签头。

use super::ops::{Linear, gelu};
use super::safetensors::WeightFile;

/// `LoRADelta`：dense_a (hid, rank, labels) + dense_b (rank, hid, labels)。
///
/// einsum 链 `bsh,hil->bsil`（其中 hil = hrl,ril->hil）拆成两步，
/// 避免物化 hid*hid*labels 的中间张量（基本句素性 48 标签时 1.1 GB）。
#[derive(Debug, Clone)]
pub(in crate::forward) struct LoraDelta {
    a: Vec<f32>,
    b: Vec<f32>,
    hid: usize,
    rank: usize,
    labels: usize,
}

impl LoraDelta {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let a = w.f32(&format!("{prefix}.dense_a"));
        let b = w.f32(&format!("{prefix}.dense_b"));
        let shape = w.shape(&format!("{prefix}.dense_a"));
        let hid = shape[0];
        let rank = shape[1];
        let labels = shape[2];
        Self {
            a,
            b,
            hid,
            rank,
            labels,
        }
    }

    /// `tmp[s, r, l] = Σ_h pooled[s, h]·A[h, r, l]`，返回 (rows*rank*labels) 平坦缓冲。
    pub(in crate::forward) fn compress(&self, pooled: &[Vec<f32>]) -> Vec<f32> {
        let rows = pooled.len();
        let mut out = vec![0.0f32; rows * self.rank * self.labels];
        for (s, x) in pooled.iter().enumerate() {
            for (h, &xh) in x.iter().enumerate() {
                if xh == 0.0 {
                    continue;
                }
                let a_base = h * self.rank * self.labels;
                let o_base = s * self.rank * self.labels;
                for i in 0..self.rank * self.labels {
                    out[o_base + i] += xh * self.a[a_base + i];
                }
            }
        }
        out
    }

    /// `out[s, i, l] = Σ_r tmp[s, r, l]·B[r, i, l]`，返回 (rows*hid*labels) 平坦缓冲。
    pub(in crate::forward) fn expand(&self, tmp: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * self.hid * self.labels];
        for s in 0..rows {
            for r in 0..self.rank {
                for l in 0..self.labels {
                    let t = tmp[s * self.rank * self.labels + r * self.labels + l];
                    if t == 0.0 {
                        continue;
                    }
                    let b_base = r * self.hid * self.labels;
                    let o_base = s * self.hid * self.labels;
                    // 只累加同一个标签通道 l 的那一列：B 的形状是 (rank, hid, labels)，
                    //   下标 = r*hid*labels + i*labels + l。写成一整段 hid*labels 的连续累加
                    //   会把 tmp[s,r,l] 摊到**全部**标签上，两个标签张量交叉污染。
                    for i in 0..self.hid {
                        let k = i * self.labels + l;
                        out[o_base + k] += t * self.b[b_base + k];
                    }
                }
            }
        }
        out
    }
}

/// `LoRASequenceMultiLabelingHead`：dense + LoRA delta，过 GELU，再过 classifier 与 sigmoid。
#[derive(Debug, Clone)]
pub struct LoraMultiLabelHead {
    dense: Linear,
    delta: LoraDelta,
    classifier_weight: Vec<f32>, // (hid, labels)
    classifier_bias: Vec<f32>,   // (labels,)
    labels: usize,
}

impl LoraMultiLabelHead {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let dense = Linear::from_weights(w, &format!("{prefix}.dense"));
        let delta = LoraDelta::from_weights(w, &format!("{prefix}.delta"));
        let classifier_weight = w.f32(&format!("{prefix}.classifier_weight"));
        let classifier_bias = w.f32(&format!("{prefix}.classifier_bias"));
        let shape = w.shape(&format!("{prefix}.classifier_weight"));
        let labels = shape[1];
        Self {
            dense,
            delta,
            classifier_weight,
            classifier_bias,
            labels,
        }
    }

    /// 输入 (rows, hid)，输出 (rows, labels) 的 sigmoid 概率。
    ///
    /// 不物化 `expand` 的 (rows*hid*labels) 中间量。
    ///   基本句素性头 rows=256、hid=768、labels=64 时约 1258 万 float（50 MB），
    ///   而它紧接着就被逐元素读走——写一块再读一块纯属浪费。
    ///   这里改成：对每个标签 l，先把该标签的 (hid) 维 delta 向量取出来，
    ///   逐 h 累加 GELU(dense[h] + delta[h,l])·classifier[h,l]。
    ///   与「先 expand 再逐元素读」的算术完全相同，只是不落那块内存。
    pub fn probabilities(&self, pooled: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows = pooled.len();
        let dense_out = self.dense.forward_rows(pooled);
        let tmp = self.delta.compress(pooled);
        let d = &self.delta;
        let mut out = Vec::with_capacity(rows);
        for s in 0..rows {
            let mut probs = Vec::with_capacity(self.labels);
            let tmp_s = &tmp[s * d.rank * d.labels..(s + 1) * d.rank * d.labels];
            let dense_s = &dense_out[s];
            for l in 0..self.labels {
                // delta[s, h, l] = Σ_r tmp[s, r, l] · B[r, h, l]
                let mut acc = self.classifier_bias[l];
                for (h, &dense_h) in dense_s.iter().enumerate() {
                    let mut delta = 0.0f32;
                    for r in 0..d.rank {
                        delta +=
                            tmp_s[r * d.labels + l] * d.b[r * d.hid * d.labels + h * d.labels + l];
                    }
                    let activated = gelu(dense_h + delta);
                    acc += activated * self.classifier_weight[h * self.labels + l];
                }
                probs.push(1.0 / (1.0 + (-acc).exp()));
            }
            out.push(probs);
        }
        out
    }
}
