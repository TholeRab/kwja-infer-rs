//! 成对选择头（dependency/discourse）的投影与按行计算。

use super::ops::{Linear, gelu};
use super::safetensors::WeightFile;

/// `WordSelectionHead` 的投影结果：source/target 线性层各算一次，行按需取。
#[derive(Debug, Clone)]
pub struct SelectionProjection {
    hs: Vec<Vec<f32>>,
    ht: Vec<Vec<f32>>,
    output_weight: Vec<f32>, // (labels, hid) 行主序，无 bias
    /// output_weight 的转置 (hid, labels)，预存给矩阵乘用。
    weight_t: ndarray::Array2<f32>,
    labels: usize,
    hid: usize,
}

impl SelectionProjection {
    /// 第 i 行的 logits：`out[j][l] = Σ_h output_w[l,h]·GELU(hs[i,h] + ht[j,h])`。
    /// 第 i 行的全部候选（(ht.len(), labels)）。
    ///
    /// labels=1 的头（dependency）逐列标量最快：矩阵乘要为每次调用建
    ///   (cols, hid) 中间数组，那份开销超过收益。
    ///   labels>1 的头（discourse，7 类）走 row_gathered 的矩阵乘路径。
    pub fn row(&self, i: usize) -> Vec<Vec<f32>> {
        if self.labels > 1 {
            let cols: Vec<usize> = (0..self.ht.len()).collect();
            return self.row_gathered(i, &cols);
        }
        let hs_i = &self.hs[i];
        let w = &self.output_weight[..self.hid];
        let mut out = Vec::with_capacity(self.ht.len());
        for tj in &self.ht {
            let mut acc = 0.0f32;
            for h in 0..self.hid {
                acc += w[h] * gelu(hs_i[h] + tj[h]);
            }
            out.push(vec![acc]);
        }
        out
    }

    pub fn rows(&self) -> usize {
        self.ht.len()
    }

    /// 只算 `cols` 里那些目标列：`out[k][l]` 对应 `row(i)[cols[k]][l]`。
    /// 逐元素算术与整行算完完全相同（每个 logit 只依赖 (i, j) 这一对）。
    ///
    /// 三步都可矩阵乘：先按对求 GELU(hs[i]+ht[j]) 得 (cols, hid)，
    ///   再与 output_weightᵀ (hid, labels) 相乘。GELU 的输入**不依赖 label**
    ///   （GELU 的输入 hidden 只由 hs/ht 决定），所以每个 (i,j) 对只需
    ///   hid 次 GELU，然后一次矩阵乘出全部 label；若 labels 在外层、
    ///   hid 在内层，这 hid 次 GELU 会重算 labels 遍。
    pub fn row_gathered(&self, i: usize, cols: &[usize]) -> Vec<Vec<f32>> {
        let rows = cols.len();
        let mut act = ndarray::Array2::<f32>::zeros((rows, self.hid));
        for (k, &j) in cols.iter().enumerate() {
            let tj = &self.ht[j];
            let hs_i = &self.hs[i];
            for h in 0..self.hid {
                act[[k, h]] = gelu(hs_i[h] + tj[h]);
            }
        }
        let out = act.dot(&self.weight_t);
        out.rows().into_iter().map(|r| r.to_vec()).collect()
    }

    /// [`Self::row_gathered`] 的再裁剪：只算 `channels` 里那些标签通道，
    /// `out[k][c]` 对应 `row(i)[cols[k]][channels[c]]`（cohesion 按任务裁剪用）。
    pub fn row_gathered_channels(
        &self,
        i: usize,
        cols: &[usize],
        channels: &[usize],
    ) -> Vec<Vec<f32>> {
        cols.iter()
            .map(|&j| {
                let tj = &self.ht[j];
                channels
                    .iter()
                    .map(|&l| {
                        let w = &self.output_weight[l * self.hid..(l + 1) * self.hid];
                        let mut acc = 0.0f32;
                        for h in 0..self.hid {
                            acc += w[h] * gelu(self.hs[i][h] + tj[h]);
                        }
                        acc
                    })
                    .collect()
            })
            .collect()
    }

    /// GPU 路径：source/target 两个线性层的结果由 ORT 算好送来，这里只装配。
    pub fn from_projected(head: &SelectionHead, hs: Vec<Vec<f32>>, ht: Vec<Vec<f32>>) -> Self {
        let weight_t =
            ndarray::Array2::from_shape_vec((head.labels, head.hid), head.output_weight.clone())
                .expect("输出层权重整形失败")
                .t()
                .to_owned();
        Self {
            hs,
            ht,
            output_weight: head.output_weight.clone(),
            weight_t,
            labels: head.labels,
            hid: head.hid,
        }
    }
}

/// `WordSelectionHead`：l_source／l_target 两个线性层 + 无 bias 的输出层。
#[derive(Debug, Clone)]
pub struct SelectionHead {
    l_source: Linear,
    l_target: Linear,
    output_weight: Vec<f32>,
    labels: usize,
    hid: usize,
}

impl SelectionHead {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let l_source = Linear::from_weights(w, &format!("{prefix}.l_source"));
        let l_target = Linear::from_weights(w, &format!("{prefix}.l_target"));
        let output_weight = w.f32(&format!("{prefix}.output_layer.weight"));
        let shape = w.shape(&format!("{prefix}.output_layer.weight"));
        let labels = shape[0];
        let hid = shape[1];
        Self {
            l_source,
            l_target,
            output_weight,
            labels,
            hid,
        }
    }

    pub fn project(&self, pooled: &[Vec<f32>]) -> SelectionProjection {
        let hs = self.l_source.forward_rows(pooled);
        let ht = self.l_target.forward_rows(pooled);
        let weight_t =
            ndarray::Array2::from_shape_vec((self.labels, self.hid), self.output_weight.clone())
                .expect("输出层权重整形失败")
                .t()
                .to_owned();
        SelectionProjection {
            hs,
            ht,
            output_weight: self.output_weight.clone(),
            weight_t,
            labels: self.labels,
            hid: self.hid,
        }
    }
}
