//! 公版语料冒烟测试：青空文库《吾輩は猫である》能正常跑通整条链。
//!
//! ## 验什么、不验什么
//!
//! **验**：
//!   1. 语料能被 char 阶段接受（RAW → Juman++，不 panic、不产出空）
//!   2. word 阶段能产出结构合法的 KNP（S-ID / * / + / EOS 齐全）
//!   3. **设备形态正确**——要求全 GPU 时，编码器与 13 个头都在 GPU
//!
//! **不验**：与 Python 参照逐字节一致。**刻意不验**——那份参照只对 第三方语料
//! 语料建过基线，青空文库这套没有；而且参照侧自身已知有偏差
//! （见事实账「c-07-49 词典命中」「c-41-78 阈值翻转」）。
//!
//! ## 语料来源与许可
//!
//! 青空文库 图书记号 789《吾輩は猫である》，作者 夏目漱石（1916 年逝世，著作权消灭）。
//! **这是本仓唯一可以随代码公开的语料**。另一套 第三方语料 语料著作权属作者，
//! 只放参考实现仓，不随本仓分发。
//!
//! 生成脚本：`参考实现脚本/aozora_to_corpus.py`（HTML → 纯文本，剥 ruby）。
//!
//! ## 设备形态（与 cli 同口径）
//!
//! | 环境变量 | 形态 |
//! | --- | --- |
//! | `KWJA_DEVICE` 不设或 `cuda` ＋ `KWJA_BACKEND` 不设或 `cublas` | 全 GPU（cuBLAS 头） |
//! | `KWJA_BACKEND=ort` | 全 GPU（ORT 融合图头） |
//! | `KWJA_DEVICE=cpu` | 全 CPU（头也留在 CPU） |
//!
//! **★混跑是被禁止的**：编码器在 GPU、头在 CPU 这种状态， 我第一版
//! 测试就写成了这样——不报错、产出也对，只是慢一倍，靠看日志才发现。
//! 现在 `assert_device_shape()` 会当场拦住。

use std::path::PathBuf;

use kwja_encoder::ort_backend::Device;
use kwja_encoder::{Encoder, make_backend};
use kwja_normalize::nfkc::normalize_text;
use kwja_pipeline::char_stage::CharStage;
use kwja_pipeline::word_predict::WordPredictor;

fn repo_path(rel: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}

fn corpus_dir() -> PathBuf {
    repo_path("crates/pipeline/tests/corpus-aozora")
}

fn onnx_dir() -> PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| repo_path("assets/models").to_string_lossy().into_owned())
        .into()
}

/// 期望的设备形态。
#[derive(Debug, PartialEq)]
enum Shape {
    /// 编码器与头都在 GPU。
    AllGpu,
    /// 编码器与头都在 CPU。
    AllCpu,
}

/// 按环境变量决定目标形态，与 cli 同口径。
fn target_shape() -> Shape {
    if std::env::var("KWJA_DEVICE").unwrap_or_default() == "cpu" {
        Shape::AllCpu
    } else {
        Shape::AllGpu
    }
}

fn backend() -> Box<dyn Encoder> {
    let dir = onnx_dir();
    let device = match target_shape() {
        Shape::AllCpu => Device::Cpu,
        Shape::AllGpu => Device::Cuda,
    };
    let backend = make_backend(
        dir.join("char_dynamic.onnx"),
        dir.join("word_encoder7.onnx"),
        device,
    )
    .expect("编码后端创建失败");

    // 断言 EP 真到位：EP 起不来时 ORT 会静默回落 CPU。
    let providers = backend.active_providers();
    let has_cuda = providers.iter().any(|p| p.contains("CUDA"));
    match target_shape() {
        Shape::AllGpu => assert!(
            has_cuda,
            "要求 CUDA，active_providers 却不含：{providers:?}（静默回落 CPU？）"
        ),
        Shape::AllCpu => assert!(
            !has_cuda,
            "要求纯 CPU，active_providers 却含 CUDA：{providers:?}"
        ),
    }
    backend
}

/// 头前向接线，与 cli 完全一致。
fn wire_heads(predictor: &mut WordPredictor, encoder: &dyn Encoder) {
    if target_shape() == Shape::AllCpu {
        eprintln!("头前向：CPU（KWJA_DEVICE=cpu）");
        return;
    }

    let models = repo_path("assets/models");
    let kind = std::env::var("KWJA_BACKEND").unwrap_or_default();
    if kind.is_empty() || kind == "cublas" {
        match kwja_heads::cublas_heads::CublasHeads::load(&models) {
            Ok(Some(gpu)) => {
                predictor
                    .set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Cublas(gpu)));
                eprintln!("头前向：cuBLAS 直调");
            }
            Ok(None) => panic!("找不到 models/cublas 权重，cuBLAS 头无法接线"),
            Err(e) => panic!("加载 cuBLAS 权重失败：{e}"),
        }
    } else {
        match kwja_heads::gpu_heads::GpuHeads::load(&models, Device::Cuda) {
            Ok(Some(gpu)) => {
                predictor.set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Ort(gpu)));
                eprintln!("头前向：ORT CUDA");
            }
            Ok(None) => panic!("找不到 heads_fused.onnx / dep_type.onnx"),
            Err(e) => panic!("加载头 ONNX 失败：{e}"),
        }
    }
    let _ = encoder;
}

/// ★**形态闸**：确认编码器与头在同一形态上，不允许混跑。
///
/// 混跑（GPU 编码 + CPU 头）不报错、产出也对，只是慢—— 我第一版测试
/// 正是这样写错的。这个断言把「静默降级」变成当场失败。
fn assert_device_shape(predictor: &WordPredictor, encoder: &dyn Encoder) {
    match target_shape() {
        Shape::AllGpu => {
            assert!(
                predictor.head_device().is_some(),
                "要求全 GPU，但头前向留在 CPU（混跑！）——检查是否漏了 wire_heads"
            );
            assert!(
                encoder.active_providers().iter().any(|p| p.contains("CUDA")),
                "要求全 GPU，但编码器不在 CUDA 上"
            );
            eprintln!("形态：全 GPU（头={:?}）", predictor.head_device());
        }
        Shape::AllCpu => {
            assert!(
                predictor.head_device().is_none(),
                "要求全 CPU，但头前向在 GPU 上：{:?}",
                predictor.head_device()
            );
            assert!(
                !encoder.active_providers().iter().any(|p| p.contains("CUDA")),
                "要求全 CPU，但编码器在 CUDA 上"
            );
            eprintln!("形态：全 CPU");
        }
    }
}

fn char_stage() -> CharStage {
    let snapshots = repo_path("assets/tokenizers")
        .join("models--ku-nlp--deberta-v2-base-japanese-char-wwm")
        .join("snapshots");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .expect("char-wwm snapshots 目录不存在")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    hits.sort();
    assert_eq!(hits.len(), 1, "char-wwm snapshots 不唯一：{hits:?}");
    CharStage::new(hits.remove(0).join("vocab.txt"), 512).expect("CharStage 构建失败")
}

/// 读公版语料全部章节；缺失返回 None（跳过）。
fn aozora_chapters() -> Option<Vec<(String, String)>> {
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!("跳过：公版语料目录不存在（{dir:?}）");
        return None;
    }
    let mut out = Vec::new();
    for k in 1..=11 {
        let p = dir.join(format!("bk-CH-{k:04}.txt"));
        match std::fs::read_to_string(&p) {
            Ok(t) => out.push((format!("CH-{k:04}"), t)),
            Err(_) => {
                eprintln!("跳过：公版语料缺 {p:?}");
                return None;
            }
        }
    }
    Some(out)
}

/// 检查 KNP 结构合法性。返回问题列表（空 = 合法）。
fn check_knp_structure(knp: &str, what: &str) -> Vec<String> {
    let mut bad = Vec::new();
    let lines: Vec<&str> = knp.lines().collect();
    if lines.is_empty() {
        bad.push(format!("{what}: 产出为空"));
        return bad;
    }
    let n_sid = lines.iter().filter(|l| l.starts_with("# S-ID:")).count();
    let n_bp = lines.iter().filter(|l| l.starts_with("+ ")).count();
    let n_ph = lines.iter().filter(|l| l.starts_with("* ")).count();
    let n_morph = lines
        .iter()
        .filter(|l| !l.is_empty() && !l.starts_with(['#', '*', '+']))
        .count();
    let n_eos = lines.iter().filter(|l| l.trim() == "EOS").count();
    if n_sid == 0 {
        bad.push(format!("{what}: 没有 # S-ID: 行（分句失败？）"));
    }
    if n_bp == 0 {
        bad.push(format!("{what}: 没有 + 行（基本句切分失败？）"));
    }
    if n_ph == 0 {
        bad.push(format!("{what}: 没有 * 行（文节切分失败？）"));
    }
    if n_eos != n_sid {
        bad.push(format!("{what}: EOS {n_eos} 个 != S-ID {n_sid} 个"));
    }
    if n_morph < n_bp {
        bad.push(format!("{what}: 词素 {n_morph} 少于基本句 {n_bp}"));
    }
    if lines.iter().any(|l| l.is_empty()) {
        bad.push(format!("{what}: 产出含空行"));
    }
    bad
}

/// 第 1 章全链路：跑通 + 结构合法 + 形态正确。
///
/// 第 1 章最短（33 段），秒级返回，适合当常规冒烟。
#[test]
fn aozora_first_chapter_runs() {
    let Some(chapters) = aozora_chapters() else {
        return;
    };
    let (id, raw) = &chapters[0];
    let normalized = normalize_text(raw);

    let mut encoder = backend();
    let mut predictor = WordPredictor::new(&repo_path("assets"), 256, 2).expect("predictor");
    wire_heads(&mut predictor, encoder.as_ref());
    assert_device_shape(&predictor, encoder.as_ref());

    let juman = char_stage()
        .run(std::slice::from_ref(&normalized), Some("c"), encoder.as_mut())
        .expect("char 阶段失败");
    assert!(juman.contains("# S-ID:"), "{id}: char 阶段没产出 S-ID 行");

    let knp = predictor
        .predict_text(encoder.as_mut(), &juman)
        .expect("word 阶段失败");

    let bad = check_knp_structure(&knp, id);
    assert!(bad.is_empty(), "结构检查未过：\n{}", bad.join("\n"));
    eprintln!("{id}: {} 行 KNP，结构合法", knp.lines().count());
}

/// 全 11 章：逐章跑通，验证长章节（第 11 章 668 段）不崩。
///
/// 比单章慢（约 1 分钟），但能确认不是只有第一章碰巧能跑。
#[test]
fn aozora_all_chapters_run() {
    let Some(chapters) = aozora_chapters() else {
        return;
    };
    assert_eq!(chapters.len(), 11, "《吾輩は猫である》应有 11 章");

    let mut encoder = backend();
    let mut predictor = WordPredictor::new(&repo_path("assets"), 256, 2).expect("predictor");
    wire_heads(&mut predictor, encoder.as_ref());
    assert_device_shape(&predictor, encoder.as_ref());
    let stage = char_stage();

    let mut total = 0usize;
    for (id, raw) in &chapters {
        let normalized = normalize_text(raw);
        let juman = stage
            .run(std::slice::from_ref(&normalized), Some("c"), encoder.as_mut())
            .unwrap_or_else(|e| panic!("{id}: char 阶段失败：{e}"));
        let knp = predictor
            .predict_text(encoder.as_mut(), &juman)
            .unwrap_or_else(|e| panic!("{id}: word 阶段失败：{e}"));
        let bad = check_knp_structure(&knp, id);
        assert!(bad.is_empty(), "结构检查未过：\n{}", bad.join("\n"));
        let n = knp.lines().count();
        total += n;
        eprintln!("{id}: {n} 行");
    }
    eprintln!("合计 {total} 行 KNP（{} 章）", chapters.len());
    assert!(total > 100_000, "总行数 {total} 偏少，可能中途出问题");
}
