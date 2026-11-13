//! 基本前向算子：GELU 与 Linear。

use super::safetensors::WeightFile;

/// torch `nn.GELU()` 默认（approximate='none'）。
#[inline]
pub(in crate::forward) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `nn.Linear`：weight 形状 (out, in) 行主序，`y = x·W^T + b`。
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Vec<f32>,
    /// 权重的转置副本 (in_features, out)，预存给矩阵乘用。
    /// torch 的 weight 是 (out, in)，而 `y = x·W^T` 需要 (in, out)。
    weight_t: ndarray::Array2<f32>,
    bias: Vec<f32>,
    pub(in crate::forward) out: usize,
    in_features: usize,
}

impl Linear {
    pub(in crate::forward) fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let weight = w.f32(&format!("{prefix}.weight"));
        let shape = w.shape(&format!("{prefix}.weight"));
        let bias = w.f32(&format!("{prefix}.bias"));
        let (out, in_features) = (shape[0], shape[1]);
        // 显式转置成 (in, out) 的 C 连续布局：.t() 只是视图，as_slice() 拿不到
        // 连续内存（panic「权重应连续」的根因）；这里一次性拷成连续布局。
        let mut weight_t_v = vec![0.0f32; in_features * out];
        for (c, w_row) in weight.chunks_exact(in_features).enumerate() {
            for (k, &wv) in w_row.iter().enumerate() {
                weight_t_v[k * out + c] = wv;
            }
        }
        let weight_t = ndarray::Array2::from_shape_vec((in_features, out), weight_t_v)
            .expect("权重转置整形失败");
        Self {
            weight,
            weight_t,
            bias,
            out,
            in_features,
        }
    }

    pub(in crate::forward) fn forward_row(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features, "Linear 输入维度不符");
        let mut y = self.bias.clone();
        for (o, row) in self.weight.chunks_exact(self.in_features).enumerate() {
            let mut acc = 0.0f32;
            for (xi, wi) in x.iter().zip(row) {
                acc += xi * wi;
            }
            y[o] += acc;
        }
        y
    }

    /// 整批前向：一次 (rows*in)·(in*out) 矩阵乘，而不是逐行 matvec。
    ///
    /// 逐行 matvec 时每个 token 都把整块权重（768*768*4B 约 2.25 MB）重扫一遍，
    /// 单窗 256 个 token 就要扫 256 遍。FLOPs 一模一样，差别在：矩阵乘能走
    /// 分块+SIMD，且权重只过一遍 cache。
    ///
    /// 等价性：浮点加法顺序变了，末位可能有差异，但 (a) 每个输出元素仍是同一串
    ///   乘加（只是累加顺序），(b) 下游判据是 argmax／阈值，实测 argmax 不受影响。
    pub(in crate::forward) fn forward_rows(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows = xs.len();
        if rows == 0 {
            return Vec::new();
        }
        // 输入拼成 (rows, in_features) 连续缓冲
        let mut a = Vec::with_capacity(rows * self.in_features);
        for x in xs {
            assert_eq!(x.len(), self.in_features, "Linear 输入维度不符");
            a.extend_from_slice(x);
        }
        // (rows, in) · (in, out) 交给 ndarray dot（matrixmultiply 分块内核）。
        // 自研分块 GEMM 实测不快于 ndarray dot，故用 dot；
        // 权重已修成 C 连续，dot 直接受益。
        let a = ndarray::Array2::from_shape_vec((rows, self.in_features), a).expect("输入整形失败");
        // (rows, in) · (in, out) 得 (rows, out)
        let mut y = a.dot(&self.weight_t);
        // 加 bias（原地）
        for mut row in y.rows_mut() {
            for (o, v) in row.iter_mut().enumerate() {
                *v += self.bias[o];
            }
        }
        y.rows().into_iter().map(|r| r.to_vec()).collect()
    }
}
