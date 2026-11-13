//! relation-wise 头（cohesion）的投影与按行、批量计算。

use super::lora::LoraDelta;
use super::ops::{Linear, gelu};
use super::safetensors::WeightFile;

/// `LoRARelationWiseWordSelectionHead` 的投影结果（delta 已并入 source/target）。
#[derive(Debug, Clone)]
pub struct RelationWiseProjection {
    source: Vec<f32>, // (rows, rels, hid) 平坦
    target: Vec<f32>,
    classifier: Vec<f32>, // (hid, rels)
    rels: usize,
    hid: usize,
}

impl RelationWiseProjection {
    /// 第 i 行的 logits：`out[j][rel] = Σ_h classifier[h,rel]·GELU(source[i,rel] + target[j,rel])`。
    pub fn row(&self, i: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(self.target.len() / (self.rels * self.hid));
        for j in 0..self.target.len() / (self.rels * self.hid) {
            let mut logits = vec![0.0f32; self.rels];
            for (rel, lg) in logits.iter_mut().enumerate() {
                let si = &self.source[i * self.rels * self.hid + rel * self.hid..][..self.hid];
                let tj = &self.target[j * self.rels * self.hid + rel * self.hid..][..self.hid];
                let mut acc = 0.0f32;
                for h in 0..self.hid {
                    acc += self.classifier[h * self.rels + rel] * gelu(si[h] + tj[h]);
                }
                *lg = acc;
            }
            out.push(logits);
        }
        out
    }

    /// 只算 `cols` 里那些目标列：`out[k][rel]` 对应 `row(i)[cols[k]][rel]`。
    ///
    /// GELU 的输入依赖 rel（source 与 target 都带 rel 分量），
    ///   所以不能把 GELU 提到 rel 循环外面——那样会改变结果。
    pub fn row_gathered(&self, i: usize, cols: &[usize]) -> Vec<Vec<f32>> {
        let n_rows = self.target.len() / (self.rels * self.hid);
        cols.iter()
            .map(|&j| {
                assert!(j < n_rows, "cohesion 目标列 {j} 越界（共 {n_rows}）");
                let mut logits = vec![0.0f32; self.rels];
                for (rel, lg) in logits.iter_mut().enumerate() {
                    let si = &self.source[i * self.rels * self.hid + rel * self.hid..][..self.hid];
                    let tj = &self.target[j * self.rels * self.hid + rel * self.hid..][..self.hid];
                    let mut acc = 0.0f32;
                    for h in 0..self.hid {
                        acc += self.classifier[h * self.rels + rel] * gelu(si[h] + tj[h]);
                    }
                    *lg = acc;
                }
                logits
            })
            .collect()
    }
}

impl RelationWiseProjection {
    /// 批量版 [`Self::row_gathered`]：一次算 `rows` 行 * `cols` 列的全部 logits。
    ///
    /// 与逐行调用逐位相同：每对 (i,j) 仍是
    ///   `Σ_h classifier[h,rel] · gelu(source[i,rel,h] + target[j,rel,h])`，
    ///   同一串乘加、同一顺序。改变的只是外层循环的套法——
    ///   逐行调用要重复 101 次函数调用和 101 次 source 切片，
    ///   而且每行的 j 循环彼此独立、无法合并访存。
    ///
    /// 这也是 cohesion 剪枝的落点：调用方只传目标行与候选列
    ///   （B*cand），不再算 256*256 全矩阵。
    pub fn rows_gathered(&self, rows: &[usize], cols: &[usize]) -> Vec<Vec<Vec<f32>>> {
        let n_rows = self.target.len() / (self.rels * self.hid);
        for &i in rows {
            for &j in cols {
                assert!(
                    i < n_rows && j < n_rows,
                    "cohesion 下标越界（行 {i} 列 {j}，共 {n_rows}）"
                );
            }
        }
        let mut out = Vec::with_capacity(rows.len());
        for &i in rows {
            out.push(self.row_gathered(i, cols));
        }
        out
    }

    /// GPU 路径：source/target（delta 已并入、rel 主序平坦缓冲）由 ORT 算好送来。
    pub fn from_projected(head: &LoraRelationWiseHead, source: Vec<f32>, target: Vec<f32>) -> Self {
        Self {
            source,
            target,
            classifier: head.classifier.clone(),
            rels: head.rels,
            hid: head.hid,
        }
    }
}
/// `LoRARelationWiseWordSelectionHead`。
#[derive(Debug, Clone)]
pub struct LoraRelationWiseHead {
    l_source: Linear,
    l_target: Linear,
    delta_source: LoraDelta,
    delta_target: LoraDelta,
    classifier: Vec<f32>, // (hid, rels)
    rels: usize,
    hid: usize,
}

impl LoraRelationWiseHead {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let l_source = Linear::from_weights(w, &format!("{prefix}.l_source"));
        let l_target = Linear::from_weights(w, &format!("{prefix}.l_target"));
        let delta_source = LoraDelta::from_weights(w, &format!("{prefix}.delta_source"));
        let delta_target = LoraDelta::from_weights(w, &format!("{prefix}.delta_target"));
        let classifier = w.f32(&format!("{prefix}.classifier"));
        let shape = w.shape(&format!("{prefix}.classifier"));
        let hid = shape[0];
        let rels = shape[1];
        Self {
            l_source,
            l_target,
            delta_source,
            delta_target,
            classifier,
            rels,
            hid,
        }
    }

    /// `source = h_source + delta_source_out`、`target = h_target + delta_target_out`，
    /// 返回 (rows, rels, hid) 的平坦缓冲。
    pub fn project(&self, pooled: &[Vec<f32>]) -> RelationWiseProjection {
        let rows = pooled.len();
        let hs = self.l_source.forward_rows(pooled);
        let ht = self.l_target.forward_rows(pooled);
        // delta 的 labels 维在这里是 rels
        let ts = self.delta_source.compress(pooled);
        let dso = self.delta_source.expand(&ts, rows);
        let tt = self.delta_target.compress(pooled);
        let dto = self.delta_target.expand(&tt, rows);
        let mut source = vec![0.0f32; rows * self.rels * self.hid];
        let mut target = vec![0.0f32; rows * self.rels * self.hid];
        for s in 0..rows {
            for rel in 0..self.rels {
                for h in 0..self.hid {
                    source[s * self.rels * self.hid + rel * self.hid + h] =
                        hs[s][h] + dso[s * self.hid * self.rels + h * self.rels + rel];
                    target[s * self.rels * self.hid + rel * self.hid + h] =
                        ht[s][h] + dto[s * self.hid * self.rels + h * self.rels + rel];
                }
            }
        }
        RelationWiseProjection {
            source,
            target,
            classifier: self.classifier.clone(),
            rels: self.rels,
            hid: self.hid,
        }
    }
}
