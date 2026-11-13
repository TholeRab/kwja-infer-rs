//! word 侧加载测试：ORT 能加载 word_encoder7.onnx，active_providers 含 CUDA，
//! last_hidden_state shape 为 [batch,seq,768]。

use kwja_encoder::ort_backend::{Device, OrtBackend};
use kwja_encoder::{Encoder, WordEncoderInput};

/// 仓内素材路径：从 crate 清单目录上溯找到仓根，再拼相对路径。
fn repo_path(rel: &str) -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}
fn onnx_dir() -> std::path::PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| repo_path("assets/models"))
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
fn word_model_loads_with_cuda_active() {
    let backend = backend();
    let providers = backend.active_providers();
    assert!(
        providers.iter().any(|p| p.contains("CUDA")),
        "active_providers 不含 CUDA：{providers:?}"
    );
}

#[test]
fn word_forward_output_shape_matches_onnx() {
    let mut backend = backend();
    let input = WordEncoderInput {
        input_ids: vec![0; 8],
        attention_mask: vec![1; 8],
        special_token_indices: (0..7).collect(),
        batch: 1,
        seq: 8,
    };
    let hidden = backend.run_word(&input).expect("word 前向失败");
    assert_eq!(hidden.shape(), [1, 8, 768].as_slice());
}
