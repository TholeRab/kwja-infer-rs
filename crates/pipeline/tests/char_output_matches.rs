//! char_stage 的 Juman++ 产出与 Python 参考实现逐字节一致。
//!
//! 基线 fixtures 在 tests/fixtures/（真 char 模型加 CUDA 生成，doc_id 前缀固定 c）。
//! 测试分层：先单章一致，再整书一致。

use std::path::PathBuf;

use kwja_encoder::ort_backend::{Device, OrtBackend};
use kwja_normalize::nfkc::normalize_text;
use kwja_pipeline::char_stage::CharStage;

/// 仓内素材路径：从 crate 清单目录上溯找到仓根，再拼相对路径。
fn repo_path(rel: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn onnx_dir() -> PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| repo_path("assets/models").to_string_lossy().into_owned())
        .into()
}

fn vocab_txt() -> PathBuf {
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
    hits.remove(0).join("vocab.txt")
}

fn char_stage() -> CharStage {
    CharStage::new(vocab_txt(), 512).expect("CharStage 构建失败")
}

/// 测试设备由 KWJA_DEVICE 决定：cuda（默认，断言 CUDA 真生效）、
/// cpu（纯 CPU，断言 provider 里没有 CUDA）、auto（打印实际 provider）。
/// 整书只跑 GPU——CPU 跑整书太慢。
fn backend() -> OrtBackend {
    let dir = onnx_dir();
    let dev = std::env::var("KWJA_DEVICE").unwrap_or_default();
    let want_cuda = dev.is_empty() || dev == "cuda";
    let pure_cpu = dev == "cpu";
    let device = if pure_cpu {
        Device::Cpu
    } else if want_cuda {
        Device::Cuda
    } else {
        Device::Auto
    };
    let backend = OrtBackend::new(
        dir.join("char_dynamic.onnx"),
        dir.join("word_encoder7.onnx"),
        device,
    )
    .expect("ORT 后端创建失败");
    // 断言设备真到位：EP 起不来时 ORT 会静默回落 CPU，
    // 产出虽应逐字节相同，但那就不是目标配置了。
    let providers = backend.active_providers();
    if want_cuda {
        assert!(
            providers.iter().any(|p| p.contains("CUDA")),
            "active_providers 不含 CUDA（静默回落 CPU？）：{providers:?}"
        );
    }
    if pure_cpu {
        assert!(
            !providers.iter().any(|p| p.contains("CUDA")),
            "要求纯 CPU，active_providers 却含 CUDA：{providers:?}"
        );
    }
    if dev == "auto" {
        eprintln!("active_providers：{providers:?}");
    }
    backend
}

fn corpus_texts() -> Vec<String> {
    (1..=59)
        .map(|k| {
            std::fs::read_to_string(corpus_dir().join(format!("bk-CH-{k:04}.txt")))
                .expect("语料章节缺失")
        })
        .collect()
}

/// 逐字节比对；失败时给出长度、行号与首个差异处的两侧内容，便于定位。
fn assert_bytes_equal(got: &str, expected: &str, what: &str) {
    if got == expected {
        return;
    }
    let (g, e) = (got.as_bytes(), expected.as_bytes());
    let pos = g
        .iter()
        .zip(e)
        .position(|(a, b)| a != b)
        .unwrap_or(g.len().min(e.len()));
    let line = got[..pos].matches('\n').count() + 1;
    let snippet = |s: &str| -> String {
        s[pos.min(s.len())..]
            .chars()
            .take(24)
            .collect::<String>()
            .replace('\n', "\\n")
    };
    panic!(
        "{what}与 Python 基线不一致：长度 {}/{}，首个差异在第 {line} 行字节 {pos}：\
         Rust {:?} vs Python {:?}",
        g.len(),
        e.len(),
        snippet(got),
        snippet(expected),
    );
}

#[test]
fn chapter1_matches_python_byte_for_byte() {
    let started = std::time::Instant::now();
    let texts = corpus_texts();
    let normalized: Vec<String> = texts[..1].iter().map(|t| normalize_text(t)).collect();
    let output = char_stage()
        .run(&normalized, Some("c"), &mut backend())
        .expect("char 阶段单章运行失败");
    eprintln!("单章 CH-0001 耗时 {:?}", started.elapsed());
    let expected =
        std::fs::read_to_string(fixtures_dir().join("char_stage_ch1.txt")).expect("单章基线缺失");
    assert_bytes_equal(&output, &expected, "单章 char 产出");
}

/// 分层运行前 N 章。N 小时秒级返回，用来在改动的第一时间发现回归，
/// 不必每次等到整书。
///
/// 基线必须按同一文档数单独生成：
///   doc_id 的零填充宽度是 len(str(文档数))，1 章时 c-0-1、多章时 c-00-1，
///   从整书基线切前缀对不上。
fn run_tier(n_chapters: usize, fixture: &str) {
    let started = std::time::Instant::now();
    let texts = corpus_texts();
    let normalized: Vec<String> = texts[..n_chapters].iter().map(|t| normalize_text(t)).collect();
    let output = char_stage()
        .run(&normalized, Some("c"), &mut backend())
        .expect("char 阶段分级运行失败");
    eprintln!("{n_chapters} 章 耗时 {:?}", started.elapsed());
    let expected = std::fs::read_to_string(fixtures_dir().join(fixture)).expect("分级基线缺失");
    assert_bytes_equal(&output, &expected, &format!("{n_chapters} 章 char 产出"));
}

#[test]
fn three_chapters_match_python_byte_for_byte() {
    run_tier(3, "char_stage_3ch.txt");
}

#[test]
fn six_chapters_match_python_byte_for_byte() {
    run_tier(6, "char_stage_6ch.txt");
}

/// 纯 CPU 形态跑前 10 章（KWJA_DEVICE=cpu 跑这个，不跑整书）。
/// 10 章的 doc_id 零填充宽度与整书相同（都是 2 位），基线可从整书切出。
#[test]
fn ten_chapters_match_python_byte_for_byte() {
    run_tier(10, "char_stage_10ch.txt");
}

#[test]
fn whole_book_matches_python_byte_for_byte() {
    let started = std::time::Instant::now();
    let texts = corpus_texts();
    let normalized: Vec<String> = texts.iter().map(|t| normalize_text(t)).collect();
    let output = char_stage()
        .run(&normalized, Some("c"), &mut backend())
        .expect("char 阶段整书运行失败");
    eprintln!("整书耗时 {:?}", started.elapsed());
    let expected =
        std::fs::read_to_string(fixtures_dir().join("char_stage_all.txt")).expect("整书基线缺失");
    assert_bytes_equal(&output, &expected, "整书 char 产出");
}
