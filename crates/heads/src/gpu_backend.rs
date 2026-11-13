//! 头前向的 GPU 后端选择——ORT 融合图（E1 主线）或 cuBLAS 直调（E5 分支）。
//!
//! 两个后端的 `forward_fused` / `dep_type_argmax` 输出口径一致（FusedHeadsOutput），
//! 调用方（word_predict）不感知差异。默认仍是 ORT；`KWJA_BACKEND=cublas` 切本枚举的
//! Cublas 分支（cli/main.rs 里选）。

use crate::cublas_heads::CublasHeads;
use crate::gpu_heads::{FusedHeadsOutput, GpuHeads};

pub enum GpuHeadsBackend {
    Ort(GpuHeads),
    Cublas(CublasHeads),
}

impl GpuHeadsBackend {
    pub fn forward_fused(
        &self,
        token_rows: &[Vec<f32>],
        pooled_rows: &[Vec<f32>],
    ) -> Result<FusedHeadsOutput, String> {
        match self {
            Self::Ort(g) => g.forward_fused(token_rows, pooled_rows),
            Self::Cublas(g) => g.forward_fused(token_rows, pooled_rows),
        }
    }

    pub fn dep_type_argmax(
        &self,
        pairs: &[(usize, usize)],
        pooled: &[Vec<f32>],
    ) -> Result<Vec<usize>, String> {
        match self {
            Self::Ort(g) => g.dep_type_argmax(pairs, pooled),
            Self::Cublas(g) => g.dep_type_argmax(pairs, pooled),
        }
    }
}
