//! word 阶段（13 头、组装、KNP 输出）与 Python 参考实现逐字节一致。
//!
//! 基线 fixtures 由真 word 模型加 CUDA 生成，输入是本仓 char 阶段的产出 fixture。
//! 对齐口径：先单章一致，再整书一致。
//! 头前向按 KWJA_BACKEND 接线（cuBLAS 直调或 ORT 融合图），与 cli 的运行配置一致。

use std::path::PathBuf;

use kwja_encoder::ort_backend::Device;
use kwja_encoder::{Encoder, make_backend};
use kwja_pipeline::word_predict::WordPredictor;

/// 仓内素材路径：从 crate 清单目录上溯找到仓根，再拼相对路径。
fn repo_path(rel: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}

fn fixtures_dir() -> PathBuf {
    repo_path("crates/pipeline/tests/fixtures")
}

fn onnx_dir() -> PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| repo_path("assets/models").to_string_lossy().into_owned())
        .into()
}

/// 测试设备由 KWJA_DEVICE 决定：cuda（默认，断言 CUDA 真生效）／cpu
///（**纯 CPU**，断言 provider 里没有 CUDA）／auto（打印实际 provider）。
/// 纯 CPU 形态只跑到 10 章；整书只跑 GPU，CPU 跑整书太慢。
fn backend() -> Box<dyn Encoder> {
    let dir = onnx_dir();
    let dev = std::env::var("KWJA_DEVICE").unwrap_or_default();
    let backend_kind = std::env::var("KWJA_BACKEND").unwrap_or_default();
    let rust_backend = backend_kind == "tract" || backend_kind == "rust";
    let want_cuda = dev.is_empty() || dev == "cuda";
    let pure_cpu = dev == "cpu";
    let device = if pure_cpu {
        Device::Cpu
    } else if want_cuda {
        Device::Cuda
    } else {
        Device::Auto
    };
    let backend = make_backend(
        dir.join("char_dynamic.onnx"),
        dir.join("word_encoder7.onnx"),
        device,
    )
    .expect("编码后端创建失败");
    // 断言设备真到位：EP 起不来时 ORT 会静默回落 CPU，
    // 产出虽应逐字节相同，但那就不是目标配置了。
    // tract 后端恒为 CPU 单线程，不做 EP 断言。
    let providers = backend.active_providers();
    if !rust_backend && want_cuda {
        assert!(
            providers.iter().any(|p| p.contains("CUDA")),
            "active_providers 不含 CUDA（静默回落 CPU？）：{providers:?}"
        );
    }
    if !rust_backend && pure_cpu {
        assert!(
            !providers.iter().any(|p| p.contains("CUDA")),
            "要求纯 CPU，active_providers 却含 CUDA：{providers:?}"
        );
    }
    if dev == "auto" || rust_backend {
        eprintln!("active_providers：{providers:?}");
    }
    backend
}

fn predictor() -> WordPredictor {
    WordPredictor::new(&repo_path("assets"), 256, 2).expect("WordPredictor 构建失败")
}

/// 逐字节比对；失败时给出长度、行号与首个差异处的两侧内容，便于定位。
fn assert_bytes_equal(got: &str, expected: &str, what: &str) {
    if got == expected {
        return;
    }
    let got_lines: Vec<&str> = got.lines().collect();
    let exp_lines: Vec<&str> = expected.lines().collect();
    let mut first_diff = None;
    for (i, (g, e)) in got_lines.iter().zip(&exp_lines).enumerate() {
        if g != e {
            first_diff = Some((i + 1, *g, *e));
            break;
        }
    }
    panic!(
        "{what} 不一致：got {} 行 / expected {} 行；首个差异 = {:?}",
        got_lines.len(),
        exp_lines.len(),
        first_diff.or(if got_lines.len() != exp_lines.len() {
            Some((exp_lines.len() + 1, "<无>", "<多出的行>"))
        } else {
            None
        })
    );
}

/// 头前向接线与 cli 相同：KWJA_BACKEND=cublas 走 cuBLAS 直调，
/// 其余走 ORT 融合图；仅 CUDA 生效时启用，权重缺失时留在 CPU。
fn wire_gpu_heads(predictor: &mut WordPredictor, has_cuda: bool) {
    if !has_cuda {
        return;
    }
    let models = repo_path("assets/models");
    let use_cublas = std::env::var("KWJA_BACKEND").unwrap_or_default() == "cublas";
    if use_cublas {
        match kwja_heads::cublas_heads::CublasHeads::load(&models) {
            Ok(Some(gpu)) => predictor
                .set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Cublas(gpu))),
            Ok(None) => eprintln!("头前向：未找到 models/cublas 权重，留在 CPU"),
            Err(e) => panic!("加载 cuBLAS 权重失败：{e}"),
        }
    } else {
        match kwja_heads::gpu_heads::GpuHeads::load(&models, Device::Cuda) {
            Ok(Some(gpu)) => predictor
                .set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Ort(gpu))),
            Ok(None) => eprintln!("头前向：未找到 heads_fused.onnx / dep_type.onnx，留在 CPU"),
            Err(e) => panic!("加载头 ONNX 失败：{e}"),
        }
    }
}

fn run_case(char_fixture: &str, knp_fixture: &str, what: &str) {
    let char_text =
        std::fs::read_to_string(fixtures_dir().join(char_fixture)).expect("char fixture 缺失");
    let expected = std::fs::read_to_string(fixtures_dir().join(knp_fixture)).expect("KNP 基线缺失");
    let mut predictor = predictor();
    let mut encoder = backend();
    let has_cuda = encoder.active_providers().iter().any(|p| p.contains("CUDA"));
    wire_gpu_heads(&mut predictor, has_cuda);
    let got = predictor
        .predict_text(encoder.as_mut(), &char_text)
        .expect("word 阶段推理失败");
    // 实际输出落盘，便于 diff 定位（失败与成功都留档）
    let dump =
        std::env::temp_dir().join(format!("kwja_got_{}", knp_fixture.replace(".txt", ".knp")));
    std::fs::write(&dump, &got).expect("实际输出写盘失败");
    eprintln!("实际输出已写入 {}", dump.display());
    assert_bytes_equal(&got, &expected, what);
}

#[test]
fn word_knp_ch1_matches() {
    run_case("char_stage_ch1.txt", "word_knp_ch1.txt", "第 1 章 KNP");
}

/// 分层测试：3 章 / 6 章，秒级到分钟级，
/// 用来在改动的第一时间发现回归，不必每次等到整书。
#[test]
fn word_knp_three_chapters_match() {
    run_case("char_stage_3ch.txt", "word_knp_3ch.txt", "3 章 KNP");
}

#[test]
fn word_knp_six_chapters_match() {
    run_case("char_stage_6ch.txt", "word_knp_6ch.txt", "6 章 KNP");
}

/// 纯 CPU 形态跑前 10 章（KWJA_DEVICE=cpu 跑这个，不跑整书）。
#[test]
fn word_knp_ten_chapters_match() {
    run_case("char_stage_10ch.txt", "word_knp_10ch.txt", "10 章 KNP");
}

#[test]
fn word_knp_whole_book_matches() {
    run_case("char_stage_all.txt", "word_knp_all.txt", "整书 KNP");
}
