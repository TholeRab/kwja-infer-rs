//! candle 纯 Rust 编码后端（路线 A，）：DeBERTa-v2 前向走本 crate
//! 内改造的 debertav2 模块（参考 candle-transformers 0.9.2 的同名实现），
//! matmul 走 gemm（纯 Rust、AVX2 运行时分发），不依赖 ORT 与任何外部 DLL。
//!
//! - 权重：_cache/kwja/v2.4 的 ckpt 经 tools/convert_ckpt_to_safetensors.py
//!   转出的 safetensors，与 ONNX 模型同目录；config 用 HF 快照里的 config.json。
//! - 多线程：单次前向单线程（gemm），并行来自 ParallelEncoder 的按窗 rayon
//!   并发——与解码/合并同一个池，全流程统一线程版图。
//! - char 模型 = 同一 DeBERTa 底座＋三个 SequenceLabelingHead
//!   （Linear→GELU→Linear，见 代码/src/kwja/modules/components/head.py:36）。

use std::path::{Path, PathBuf};

use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use ndarray::Array3;

use crate::debertav2::{Config, DebertaV2Model};
use crate::{
    CharEncoderInput, CharLogits, Encoder, EncoderError, ParallelEncoder, WordEncoderInput,
};

const CHAR_HEAD_HIDDEN: usize = 768;

pub struct CandleBackend {
    word: DebertaV2Model,
    char: DebertaV2Model,
    char_heads: CharHeads,
    device: Device,
}

/// char 的三个序列标注头（head.py 的 SequenceLabelingHead：
/// Linear(hidden,hidden)→GELU→Dropout→Linear(hidden,num_labels)，推理期 Dropout 恒等）。
struct CharHeads {
    sent_segmentation: (candle_nn::Linear, candle_nn::Linear),
    word_segmentation: (candle_nn::Linear, candle_nn::Linear),
    word_norm_op: (candle_nn::Linear, candle_nn::Linear),
}

fn load_head(
    vb: &VarBuilder,
    name: &str,
    hidden: usize,
    labels: usize,
) -> Result<(candle_nn::Linear, candle_nn::Linear), EncoderError> {
    let first = candle_nn::linear(hidden, hidden, vb.pp(format!("{name}.0")))
        .map_err(candle_err(name))?;
    let second = candle_nn::linear(hidden, labels, vb.pp(format!("{name}.3")))
        .map_err(candle_err(name))?;
    Ok((first, second))
}

impl CharHeads {
    fn load(vb: &VarBuilder, hidden: usize) -> Result<Self, EncoderError> {
        Ok(Self {
            sent_segmentation: load_head(vb, "sent_segmentation_tagger", hidden, 2)?,
            word_segmentation: load_head(vb, "word_segmentation_tagger", hidden, 2)?,
            word_norm_op: load_head(vb, "word_norm_op_tagger", hidden, 6)?,
        })
    }
}

fn candle_err(what: &str) -> impl Fn(candle::Error) -> EncoderError + '_ {
    let what = what.to_string();
    move |e| EncoderError::Session(format!("{what}: {e:?}"))
}

fn i64_tensor(data: &[i64], dims: (usize, usize), device: &Device) -> Result<Tensor, EncoderError> {
    Tensor::new(data, device)
        .and_then(|t| t.reshape(dims))
        .map_err(|e| EncoderError::Shape(format!("输入张量 {dims:?} 创建失败：{e:?}")))
}

fn to_array3(tensor: Tensor, name: &str, d0: usize, d1: usize, d2: usize) -> Result<Array3<f32>, EncoderError> {
    let shape = tensor.shape();
    if shape.dims() != [d0, d1, d2] {
        return Err(EncoderError::Shape(format!(
            "{name} 形状应为 ({d0},{d1},{d2})，实际 {shape:?}"
        )));
    }
    let data = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| EncoderError::Session(format!("{name} 提取失败：{e:?}")))?;
    Array3::from_shape_vec((d0, d1, d2), data)
        .map_err(|e| EncoderError::Shape(format!("{name} 数据长度不符：{e}")))
}

/// 从 HF hub 缓存目录里找某模型的 config.json（取第一个含 config.json 的快照）。
fn snapshot_config(hub: &Path, model_dir_name: &str) -> Result<PathBuf, EncoderError> {
    let snapshots = hub.join(model_dir_name).join("snapshots");
    let mut entries = std::fs::read_dir(&snapshots)
        .map_err(|e| EncoderError::Session(format!("读 {} 失败：{e}", snapshots.display())))?;
    while let Some(entry) = entries.next().transpose().map_err(|e| EncoderError::Session(format!("{e}")))? {
        let candidate = entry.path().join("config.json");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(EncoderError::Session(format!(
        "{} 下没有含 config.json 的快照",
        snapshots.display()
    )))
}

fn load_config(path: &Path) -> Result<Config, EncoderError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| EncoderError::Session(format!("读 {} 失败：{e}", path.display())))?;
    serde_json::from_str::<Config>(&text)
        .map_err(|e| EncoderError::Session(format!("解析 {} 失败：{e}", path.display())))
}

fn load_model(
    safetensors_path: &Path,
    config_path: &Path,
    device: &Device,
) -> Result<DebertaV2Model, EncoderError> {
    let tensors = candle::safetensors::load(safetensors_path, device)
        .map_err(|e| EncoderError::Session(format!("加载 {} 失败：{e:?}", safetensors_path.display())))?;
    // KWJA 侧 resize_token_embeddings(vocab + 特殊 token 数)——ckpt 的词表比
    // config.json 大（word: 32000→32008）。以实际权重为准。
    let actual_vocab = tensors
        .get("embeddings.word_embeddings.weight")
        .map(|t| {
            t.dim(0)
                .map_err(|e| EncoderError::Session(format!("词表维度读取失败：{e:?}")))
        })
        .transpose()?;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let mut config = load_config(config_path)?;
    if let Some(vocab) = actual_vocab {
        config.vocab_size = vocab;
    }
    DebertaV2Model::load(vb, &config).map_err(|e| {
        EncoderError::Session(format!("DebertaV2Model 加载失败（{}）：{e:?}", safetensors_path.display()))
    })
}

impl CandleBackend {
    pub fn new(
        char_model: impl AsRef<Path>,
        word_model: impl AsRef<Path>,
    ) -> Result<Self, EncoderError> {
        let device = Device::Cpu;
        let word_onnx = word_model.as_ref();
        let char_onnx = char_model.as_ref();
        // 权重 safetensors 与 ONNX 模型同目录（tools/convert_ckpt_to_safetensors.py 生成）。
        let word_weights = word_onnx
            .parent()
            .unwrap_or(Path::new("."))
            .join("word_deberta-v2-base.safetensors");
        let char_weights = char_onnx
            .parent()
            .unwrap_or(Path::new("."))
            .join("char_deberta-v2-base-wwm.safetensors");
        // HF config 从 hub 缓存解析；缓存根 = kwja-infer-rs 的上一级 外部素材目录/hf/hub。
        let hub = word_onnx
            .ancestors()
            .nth(3)
            .map(|p| p.join("../外部素材目录/hf/hub"))
            .unwrap_or_else(|| PathBuf::from("../外部素材目录/hf/hub"));
        let hub = std::env::var("KWJA_HF_HUB").map(PathBuf::from).unwrap_or(hub);
        let word_config = snapshot_config(&hub, "models--ku-nlp--deberta-v2-base-japanese")?;
        let char_config = snapshot_config(&hub, "models--ku-nlp--deberta-v2-base-japanese-char-wwm")?;
        Ok(Self {
            word: load_model(&word_weights, &word_config, &device)?,
            char: load_model(&char_weights, &char_config, &device)?,
            char_heads: CharHeads::load(
                &VarBuilder::from_tensors(
                    candle::safetensors::load(&char_weights, &device)
                        .map_err(|e| EncoderError::Session(format!("加载 char 权重失败：{e:?}")))?,
                    DType::F32,
                    &device,
                ),
                CHAR_HEAD_HIDDEN,
            )?,
            device,
        })
    }

    fn head_forward(
        head: &(candle_nn::Linear, candle_nn::Linear),
        hidden: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        head.0
            .forward(hidden)
            .and_then(|t| t.gelu_erf())
            .and_then(|t| head.1.forward(&t))
            .map_err(|e| EncoderError::Session(format!("char 头前向失败：{e:?}")))
    }
}

impl Encoder for CandleBackend {
    fn run_char(&mut self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError> {
        let seq = input.seq;
        let ids = i64_tensor(&input.input_ids, (1, seq), &self.device)?;
        let mask = i64_tensor(&input.attention_mask, (1, seq), &self.device)?;
        let hidden = self
            .char
            .forward(&ids, None, Some(mask), None)
            .map_err(|e| EncoderError::Session(format!("char 前向失败：{e:?}")))?;
        let sent = Self::head_forward(&self.char_heads.sent_segmentation, &hidden)?;
        let word_seg = Self::head_forward(&self.char_heads.word_segmentation, &hidden)?;
        let norm_op = Self::head_forward(&self.char_heads.word_norm_op, &hidden)?;
        Ok(CharLogits {
            sent_segmentation: to_array3(sent, "sent_segmentation_logits", 1, seq, 2)?,
            word_segmentation: to_array3(word_seg, "word_segmentation_logits", 1, seq, 2)?,
            word_norm_op: to_array3(norm_op, "word_norm_op_logits", 1, seq, 6)?,
        })
    }

    fn run_word(&mut self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError> {
        let (batch, seq) = (input.batch, input.seq);
        let ids = i64_tensor(&input.input_ids, (batch, seq), &self.device)?;
        let mask = i64_tensor(&input.attention_mask, (batch, seq), &self.device)?;
        let specials = i64_tensor(&input.special_token_indices, (batch, 7), &self.device)?;
        let hidden = self
            .word
            .forward(&ids, None, Some(mask), Some(&specials))
            .map_err(|e| EncoderError::Session(format!("word 前向失败：{e:?}")))?;
        to_array3(hidden, "last_hidden_state", batch, seq, 768)
    }

    fn as_parallel(&self) -> Option<&dyn ParallelEncoder> {
        Some(self)
    }

    fn backend_name(&self) -> &str {
        "candle"
    }

    fn active_providers(&self) -> Vec<String> {
        vec!["candle-cpu".to_string()]
    }
}

impl ParallelEncoder for CandleBackend {
    fn run_word_shared(&self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError> {
        let (batch, seq) = (input.batch, input.seq);
        let ids = i64_tensor(&input.input_ids, (batch, seq), &self.device)?;
        let mask = i64_tensor(&input.attention_mask, (batch, seq), &self.device)?;
        let specials = i64_tensor(&input.special_token_indices, (batch, 7), &self.device)?;
        let hidden = self
            .word
            .forward(&ids, None, Some(mask), Some(&specials))
            .map_err(|e| EncoderError::Session(format!("word 前向失败：{e:?}")))?;
        to_array3(hidden, "last_hidden_state", batch, seq, 768)
    }

    fn run_char_shared(&self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError> {
        // &self 版本：字段访问相同（candle 前向本来就是 &self）。
        let seq = input.seq;
        let ids = i64_tensor(&input.input_ids, (1, seq), &self.device)?;
        let mask = i64_tensor(&input.attention_mask, (1, seq), &self.device)?;
        let hidden = self
            .char
            .forward(&ids, None, Some(mask), None)
            .map_err(|e| EncoderError::Session(format!("char 前向失败：{e:?}")))?;
        let sent = Self::head_forward(&self.char_heads.sent_segmentation, &hidden)?;
        let word_seg = Self::head_forward(&self.char_heads.word_segmentation, &hidden)?;
        let norm_op = Self::head_forward(&self.char_heads.word_norm_op, &hidden)?;
        Ok(CharLogits {
            sent_segmentation: to_array3(sent, "sent_segmentation_logits", 1, seq, 2)?,
            word_segmentation: to_array3(word_seg, "word_segmentation_logits", 1, seq, 2)?,
            word_norm_op: to_array3(norm_op, "word_norm_op_logits", 1, seq, 6)?,
        })
    }
}
