//! 12 个头的 ORT 前向（头也上 GPU）。
//!
//! 两张 ONNX 由 `导出Rust侧13头ONNX.py` 导出：
//! - `heads_fused.onnx`：输入 hidden (1,T,768) 与 pooled (1,M,768)，一次前向出
//!   reading/pos/subpos/conjtype/conjform 的 logits、word_feature/base_phrase_feature
//!   的概率（图内已含 sigmoid，与 Rust CPU 路径的 `probabilities` 同口径）、ne 的
//!   logits，以及 dependency/cohesion/discourse 的 source/target 投影；
//! - `dep_type.onnx`：输入 (P,1536)（主辞与候选母辞的拼接），输出 (P,4) logits。
//!
//! pairwise 头的 seq×seq 中间量（gelu(source+target) 后与 output_weight 的点积）
//! 仍留在 Rust 惰性算——那份中间量约 186MB/窗，不适合上 GPU；这里只搬最重的线性层。
//! 两个 Session 都包在 Mutex 里：解码线程以 `&self` 并发跑多个窗，而 GPU 内核本就
//! 串行，锁的等待可以忽略。TF32 已在进程级 Environment 的 provider 选项里关闭
//!（cuda_ep.rs），与 encoder 的逐字节口径一致。

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

use kwja_encoder::ort_backend::{global_environment, Device};

use crate::sequence_labeling::argmax;

/// 融合图一次前向的全部输出；各项与 CPU 路径（`forward.rs` 各头）的形状逐项一致。
#[derive(Debug, Clone)]
pub struct FusedHeadsOutput {
    /// reading 每行 argmax（首个最大）——解码侧只消费它，不必搬整张 (T,12907) 矩阵。
    pub reading_argmax: Vec<usize>,
    pub pos_logits: Vec<Vec<f32>>,      // (M, 14)
    pub subpos_logits: Vec<Vec<f32>>,   // (M, 35)
    pub conjtype_logits: Vec<Vec<f32>>, // (M, 33)
    pub conjform_logits: Vec<Vec<f32>>, // (M, 81)
    pub wf_prob: Vec<Vec<f32>>,         // (M, 5)，已含 sigmoid
    pub bpf_prob: Vec<Vec<f32>>,        // (M, 64)，已含 sigmoid
    pub ne_logits: Vec<Vec<f32>>,       // (M, 17)
    pub dep_hs: Vec<Vec<f32>>,          // (M, 768)
    pub dep_ht: Vec<Vec<f32>>,          // (M, 768)
    /// (M, 9, 768) 平坦缓冲，rel 主序——与 `RelationWiseProjection.source` 同布局。
    pub coh_source: Vec<f32>,
    pub coh_target: Vec<f32>,
    pub dis_hs: Vec<Vec<f32>>, // (M, 768)
    pub dis_ht: Vec<Vec<f32>>, // (M, 768)
}

/// 12 个头的 ORT 会话对。`load` 在模型文件缺失时返回 `Ok(None)`，调用方回退 CPU 路径。
pub struct GpuHeads {
    fused: Mutex<Session>,
    dep_type: Mutex<Session>,
}

impl GpuHeads {
    pub fn load(models_dir: &Path, device: Device) -> Result<Option<Self>, String> {
        let fused_path = models_dir.join("heads_fused.onnx");
        let dep_type_path = models_dir.join("dep_type.onnx");
        if !fused_path.is_file() || !dep_type_path.is_file() {
            return Ok(None);
        }
        let environment = global_environment(device).map_err(|e| e.to_string())?;
        let fused = load_session(environment, &fused_path)?;
        let dep_type = load_session(environment, &dep_type_path)?;
        Ok(Some(Self {
            fused: Mutex::new(fused),
            dep_type: Mutex::new(dep_type),
        }))
    }

    /// 融合图前向：token 级行集走 hidden，词素级行集（含 special 区）走 pooled。
    pub fn forward_fused(
        &self,
        token_rows: &[Vec<f32>],
        pooled_rows: &[Vec<f32>],
    ) -> Result<FusedHeadsOutput, String> {
        let hidden = rows_to_array3(token_rows)?;
        let pooled = rows_to_array3(pooled_rows)?;
        let mut session = self.fused.lock().map_err(|_| "fused 会话锁中毒".to_string())?;
        let outputs = session
            .run(ort::inputs![
                "hidden" => TensorRef::from_array_view(&hidden)
                    .map_err(|e| format!("hidden 张量创建失败：{e}"))?,
                "pooled" => TensorRef::from_array_view(&pooled)
                    .map_err(|e| format!("pooled 张量创建失败：{e}"))?,
            ])
            .map_err(|e| format!("头融合前向失败：{e}"))?;
        Ok(FusedHeadsOutput {
            reading_argmax: extract_reading_argmax(&outputs, "reading_logits")?,
            pos_logits: extract_rows(&outputs, "pos_logits")?,
            subpos_logits: extract_rows(&outputs, "subpos_logits")?,
            conjtype_logits: extract_rows(&outputs, "conjtype_logits")?,
            conjform_logits: extract_rows(&outputs, "conjform_logits")?,
            wf_prob: extract_rows(&outputs, "wf_prob")?,
            bpf_prob: extract_rows(&outputs, "bpf_prob")?,
            ne_logits: extract_rows(&outputs, "ne_logits")?,
            dep_hs: extract_rows(&outputs, "dep_source")?,
            dep_ht: extract_rows(&outputs, "dep_target")?,
            coh_source: extract_flat(&outputs, "coh_source")?,
            coh_target: extract_flat(&outputs, "coh_target")?,
            dis_hs: extract_rows(&outputs, "dis_source")?,
            dis_ht: extract_rows(&outputs, "dis_target")?,
        })
    }

    /// 依存类型：全部 (主辞行, 候选母辞行) 对拼成 (P,1536) 一次前向，逐行 argmax。
    /// 与 CPU 路径的 `PairSequenceHead::predict_pair` 同序（argmax 并列取首）。
    pub fn dep_type_argmax(
        &self,
        pairs: &[(usize, usize)],
        pooled: &[Vec<f32>],
    ) -> Result<Vec<usize>, String> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        const PAIR_DIM: usize = 1536;
        let rows = pairs.len();
        let mut flat = vec![0.0f32; rows * PAIR_DIM];
        for (k, (hg, parent)) in pairs.iter().enumerate() {
            let base = k * PAIR_DIM;
            flat[base..base + 768].copy_from_slice(&pooled[*hg]);
            flat[base + 768..base + PAIR_DIM].copy_from_slice(&pooled[*parent]);
        }
        let pairs_array = ndarray::Array2::from_shape_vec((rows, PAIR_DIM), flat)
            .map_err(|e| format!("依存对整形失败：{e}"))?;
        let mut session = self.dep_type.lock().map_err(|_| "dep_type 会话锁中毒".to_string())?;
        let outputs = session
            .run(ort::inputs![
                "pairs" => TensorRef::from_array_view(&pairs_array)
                    .map_err(|e| format!("pairs 张量创建失败：{e}"))?,
            ])
            .map_err(|e| format!("依存类型前向失败：{e}"))?;
        let logits = extract_rows(&outputs, "logits")?;
        Ok(logits.iter().map(|row| argmax(row)).collect())
    }
}

fn load_session(environment: &Environment, path: &Path) -> Result<Session, String> {
    Session::builder(environment)
        .map_err(|e| format!("创建会话构建器失败：{e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("加载模型 {} 失败：{e}", path.display()))
}

/// 行集 → (1, n, hid) 张量（融合图的输入是 3D，batch 恒为 1）。
fn rows_to_array3(rows: &[Vec<f32>]) -> Result<Array3<f32>, String> {
    let n = rows.len();
    let hid = rows.first().map(Vec::len).unwrap_or(0);
    let mut flat = Vec::with_capacity(n * hid);
    for row in rows {
        if row.len() != hid {
            return Err(format!("行集宽度不一：{hid} vs {}", row.len()));
        }
        flat.extend_from_slice(row);
    }
    Array3::from_shape_vec((1, n, hid), flat).map_err(|e| format!("行集整形失败：{e}"))
}

/// reading 的 (T,12907) logits → 每行 argmax（首个最大，与
/// [`crate::sequence_labeling::argmax`] 同口径）。直接在 ORT 的平坦缓冲上扫，
/// 不落成行向量——那份中间量每窗约 13MB，解码侧本就只消费 argmax。
fn extract_reading_argmax(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<usize>, String> {
    let value = outputs.get(name).ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    let (n, c) = match shape {
        [n, c] => (*n as usize, *c as usize),
        [1, n, c] => (*n as usize, *c as usize),
        other => return Err(format!("{name} 形状应为 [n,c] 或 [1,n,c]，实际 {other:?}")),
    };
    if data.len() != n * c {
        return Err(format!("{name} 数据长度 {} 与形状 ({n},{c}) 不符", data.len()));
    }
    Ok(data.chunks_exact(c).map(|row| argmax(row)).collect())
}

/// [n,c] 或 [1,n,c] 的 f32 输出 → 行向量集。
fn extract_rows(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let value = outputs.get(name).ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    let (n, c) = match shape {
        [n, c] => (*n as usize, *c as usize),
        [1, n, c] => (*n as usize, *c as usize),
        other => return Err(format!("{name} 形状应为 [n,c] 或 [1,n,c]，实际 {other:?}")),
    };
    if data.len() != n * c {
        return Err(format!("{name} 数据长度 {} 与形状 ({n},{c}) 不符", data.len()));
    }
    Ok(data.chunks_exact(c).map(<[f32]>::to_vec).collect())
}

/// 4D f32 输出 → 平坦缓冲（cohesion 的 (1,M,9,768) 用）。
fn extract_flat(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<f32>, String> {
    let value = outputs.get(name).ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    if shape.len() != 4 || shape[0] != 1 {
        return Err(format!("{name} 形状应为 [1,M,rel,hid]，实际 {shape:?}"));
    }
    Ok(data.to_vec())
}
