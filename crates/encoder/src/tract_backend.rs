//! tract 纯 Rust 后端（实验分支 rust-encoder-no-ort）：直接加载 ONNX，
//! 不依赖 ONNX Runtime 与任何外部 DLL。
//!
//! ★目的：验证「拿掉 ORT」的可行性（用户  提出）。真正的动机不是
//! 「算法比 ORT 快」，而是**统一线程版图**：ORT 的 CPU EP 自带 intra-op 线程池，
//! 与 rayon 线程池互相竞争同一批核；编码挪进 Rust 侧后，全流程只有一个调度域。
//!
//! - 模型文件不变（assets/models/char_dynamic.onnx / word_encoder7.onnx，含外部权重 .data）；
//!   Constant 节点已由 tools/normalize_onnx_constants.py 归一（tract 不认 value_ints）。
//! - ★char 模型的 seq 是动态维，tract 的符号形状推理在 StridedSlice（seq-1）上
//!   会死锁（「Impossible to unify Sym(seq) with Add([Sym(seq), Val(-1)])」）。
//!   对策：按**具体 (batch, seq)** 设输入事实、编译并缓存可执行计划——char 每
//!   个文档一种 seq、word 本分支恒为 (1, 256)，各建一次，之后复用。
//! - CPU 单线程确定性执行，输出可复现；性能预期低于多核 ORT，先证正确性再谈优化。
//! - ONNX 输入/输出顺序已用 onnx 包实测：
//!   word: [input_ids, attention_mask, special_token_indices] → [last_hidden_state]
//!   char: [input_ids, attention_mask] → [sent_seg_logits, word_seg_logits, word_norm_op_logits]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::Array3;
use tract_onnx::prelude::*;

use crate::{
    CharEncoderInput, CharLogits, Encoder, EncoderError, ParallelEncoder, WordEncoderInput,
};

type Runnable = TypedRunnableModel<TypedModel>;

/// tract 后端：按输入形状缓存编译产物（见模块注释的符号维问题）。
/// 缓存用 Mutex＋Arc：计划不可变（Sync），并行预备的多个线程可以同时
/// 取同一份计划各跑各的前向（这正是统一 rayon 池的前提）。
pub struct TractBackend {
    char_path: std::path::PathBuf,
    word_path: std::path::PathBuf,
    threads: usize,
    char_cache: Mutex<HashMap<usize, Arc<Runnable>>>,
    word_cache: Mutex<HashMap<(usize, usize), Arc<Runnable>>>,
}

impl TractBackend {
    /// 记下模型路径；word 本分支恒为 (1, max_seq_length) 直接建好，
    /// char 的 seq 逐文档不同，按需编译缓存。
    pub fn new(
        char_model: impl AsRef<Path>,
        word_model: impl AsRef<Path>,
    ) -> Result<Self, EncoderError> {
        // ★窗内并行：让 tract 的 matmul 内核吃到 multithread-mm 线程池，
        //   单次前向不再单线程。线程数可用 KWJA_TRACT_THREADS 覆盖，默认＝物理核数。
        let threads = std::env::var("KWJA_TRACT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
            });
        tract_linalg::multithread::set_default_executor(
            tract_linalg::multithread::Executor::multithread(threads),
        );
        let char_path = char_model.as_ref().to_path_buf();
        let word_path = word_model.as_ref().to_path_buf();
        let mut word_cache = HashMap::new();
        word_cache.insert((1, 256), Arc::new(build_word(&word_path, 1, 256, threads)?));
        Ok(Self {
            char_path,
            word_path,
            threads,
            char_cache: Mutex::new(HashMap::new()),
            word_cache: Mutex::new(word_cache),
        })
    }
}

impl TractBackend {
    fn char_runnable(&self, seq: usize) -> Result<Arc<Runnable>, EncoderError> {
        let mut cache = self.char_cache.lock().expect("char 计划缓存锁");
        if let Some(runnable) = cache.get(&seq) {
            return Ok(runnable.clone());
        }
        let runnable = Arc::new(build_char(&self.char_path, seq, self.threads)?);
        cache.insert(seq, runnable.clone());
        Ok(runnable)
    }

    fn word_runnable(&self, batch: usize, seq: usize) -> Result<Arc<Runnable>, EncoderError> {
        let mut cache = self.word_cache.lock().expect("word 计划缓存锁");
        if let Some(runnable) = cache.get(&(batch, seq)) {
            return Ok(runnable.clone());
        }
        let runnable = Arc::new(build_word(&self.word_path, batch, seq, self.threads)?);
        cache.insert((batch, seq), runnable.clone());
        Ok(runnable)
    }
}

fn i64_fact(shape: &[usize]) -> InferenceFact {
    InferenceFact::dt_shape(DatumType::I64, shape)
}

fn build_char(path: &Path, seq: usize, threads: usize) -> Result<Runnable, EncoderError> {
    let mut model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| EncoderError::Session(format!("tract 加载 {} 失败：{:?}", path.display(), e)))?;
    model
        .set_input_fact(0, i64_fact(&[1, seq]))
        .map_err(|e| EncoderError::Session(format!("char 输入事实设置失败：{e}")))?;
    model
        .set_input_fact(1, i64_fact(&[1, seq]))
        .map_err(|e| EncoderError::Session(format!("char 掩码事实设置失败：{e}")))?;
    finish(model, threads)
}

fn build_word(path: &Path, batch: usize, seq: usize, threads: usize) -> Result<Runnable, EncoderError> {
    let mut model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| EncoderError::Session(format!("tract 加载 {} 失败：{:?}", path.display(), e)))?;
    model
        .set_input_fact(0, i64_fact(&[batch, seq]))
        .map_err(|e| EncoderError::Session(format!("word 输入事实设置失败：{e}")))?;
    model
        .set_input_fact(1, i64_fact(&[batch, seq]))
        .map_err(|e| EncoderError::Session(format!("word 掩码事实设置失败：{e}")))?;
    model
        .set_input_fact(2, i64_fact(&[batch, 7]))
        .map_err(|e| EncoderError::Session(format!("word specials 事实设置失败：{e}")))?;
    finish(model, threads)
}

fn finish(model: InferenceModel, threads: usize) -> Result<Runnable, EncoderError> {
    // ★线程池必须挂在**计划**上（plan.rs 的 eval 只在 plan.executor 存在时
    //   进入 multithread_tract_scope，光设全局默认不生效——实测踩过）。
    let options = PlanOptions {
        skip_order_opt_ram: false,
        executor: Some(tract_linalg::multithread::Executor::multithread(threads)),
    };
    model
        .into_optimized()
        .and_then(|m| m.into_runnable_with_options(&options))
        .map_err(|e| EncoderError::Session(format!("tract 优化失败：{e:?}")))
}

fn tensor2(data: &[i64], rows: usize, cols: usize) -> Result<Tensor, EncoderError> {
    Tensor::from_shape(&[rows, cols], data)
        .map_err(|e| EncoderError::Shape(format!("输入张量 ({rows},{cols}) 创建失败：{e}")))
}

impl Encoder for TractBackend {
    fn run_char(&mut self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError> {
        ParallelEncoder::run_char_shared(self, input)
    }

    fn run_word(&mut self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError> {
        ParallelEncoder::run_word_shared(self, input)
    }

    fn as_parallel(&self) -> Option<&dyn ParallelEncoder> {
        Some(self)
    }

    fn backend_name(&self) -> &str {
        "tract"
    }

    fn active_providers(&self) -> Vec<String> {
        vec!["tract-cpu".to_string()]
    }
}

impl ParallelEncoder for TractBackend {
    fn run_char_shared(&self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError> {
        let seq = input.seq;
        let ids = tensor2(&input.input_ids, 1, seq)?;
        let mask = tensor2(&input.attention_mask, 1, seq)?;
        let runnable = self.char_runnable(seq)?;
        let outs = runnable
            .run(tvec![ids.into_tvalue(), mask.into_tvalue()])
            .map_err(|e| EncoderError::Session(format!("char 前向失败：{e}")))?;
        Ok(CharLogits {
            sent_segmentation: out3(&outs[0], "sent_segmentation_logits", 1, seq, 2)?,
            word_segmentation: out3(&outs[1], "word_segmentation_logits", 1, seq, 2)?,
            word_norm_op: out3(&outs[2], "word_norm_op_logits", 1, seq, 6)?,
        })
    }

    fn run_word_shared(&self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError> {
        let (batch, seq) = (input.batch, input.seq);
        let ids = tensor2(&input.input_ids, batch, seq)?;
        let mask = tensor2(&input.attention_mask, batch, seq)?;
        let specials = tensor2(&input.special_token_indices, batch, 7)?;
        let runnable = self.word_runnable(batch, seq)?;
        let outs = runnable
            .run(tvec![
                ids.into_tvalue(),
                mask.into_tvalue(),
                specials.into_tvalue()
            ])
            .map_err(|e| EncoderError::Session(format!("word 前向失败：{e}")))?;
        out3(&outs[0], "last_hidden_state", batch, seq, 768)
    }
}

fn out3(
    value: &TValue,
    name: &str,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Result<Array3<f32>, EncoderError> {
    let view = value
        .to_array_view::<f32>()
        .map_err(|e| EncoderError::Session(format!("输出 {name} 提取失败：{e}")))?;
    let shape = view.shape();
    let expected = [d0, d1, d2];
    if shape != expected.as_slice() {
        return Err(EncoderError::Shape(format!(
            "{name} 形状应为 {expected:?}，实际 {shape:?}"
        )));
    }
    let data: Vec<f32> = view.iter().copied().collect();
    Array3::from_shape_vec((d0, d1, d2), data)
        .map_err(|e| EncoderError::Shape(format!("{name} 数据长度不符：{e}")))
}
