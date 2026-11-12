//! 阶段 1 闸（char）：ORT 能加载 char_dynamic.onnx，active_providers 含 CUDA，
//! 输出 shape 与方案 §三 表一致（sent/word [1,seq,2]、word_norm_op [1,seq,6]）。

use kwja_encoder::ort_backend::{Device, OrtBackend};
use kwja_encoder::{CharEncoderInput, Encoder};

fn onnx_dir() -> std::path::PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| r"外部素材\_cache\onnx".to_string())
        .into()
}

fn backend() -> OrtBackend {
    let dir = onnx_dir();
    OrtBackend::new(
        dir.join("char_dynamic.onnx"),
        dir.join("word_encoder7.onnx"),
        Device::Cuda,
    )
    .expect("ORT 后端创建失败")
}

#[test]
fn char_model_loads_with_cuda_active() {
    let backend = backend();
    let providers = backend.active_providers();
    assert!(
        providers.iter().any(|p| p.contains("CUDA")),
        "active_providers 不含 CUDA：{providers:?}"
    );
    assert_eq!(backend.backend_name(), "ort");
}

#[test]
fn char_forward_output_shapes_match_onnx() {
    let mut backend = backend();
    let input = CharEncoderInput {
        input_ids: vec![0; 8],
        attention_mask: vec![1; 8],
        seq: 8,
    };
    let logits = backend.run_char(&input).expect("char 前向失败");
    assert_eq!(logits.sent_segmentation.shape(), [1, 8, 2].as_slice());
    assert_eq!(logits.word_segmentation.shape(), [1, 8, 2].as_slice());
    assert_eq!(logits.word_norm_op.shape(), [1, 8, 6].as_slice());
}
