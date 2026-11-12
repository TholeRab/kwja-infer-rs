//! 阶段 5 闸：char_stage 的 Juman++ 产出与 Python 扁平链**逐字节**一致。
//!
//! 基线由 `导出工具/导出Rust侧阶段5基准.py` 生成
//! （本仓 Python 实现＋真 char 模型＋CUDA，doc_id 前缀固定 "c"）。
//! 闸口径（实施方案 §四 阶段 5）：先单章一致，再整书一致。

use std::path::PathBuf;

use kwja_encoder::ort_backend::{Device, OrtBackend};
use kwja_normalize::nfkc::normalize_text;
use kwja_pipeline::char_stage::CharStage;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn onnx_dir() -> PathBuf {
    std::env::var("KWJA_ONNX_DIR")
        .unwrap_or_else(|_| r"外部素材\_cache\onnx".to_string())
        .into()
}

fn vocab_txt() -> PathBuf {
    let snapshots = PathBuf::from(r"外部素材\_cache\hf\hub")
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

fn backend() -> OrtBackend {
    let dir = onnx_dir();
    let backend = OrtBackend::new(
        dir.join("char_dynamic.onnx"),
        dir.join("word_encoder7.onnx"),
        Device::Cuda,
    )
    .expect("ORT 后端创建失败");
    // ★必须显式断言 CUDA 生效：EP 起不来时 ORT 会静默回落 CPU，
    // 产出虽应逐字节相同，但那就不是目标配置了（事实账 §六）。
    let providers = backend.active_providers();
    assert!(
        providers.iter().any(|p| p.contains("CUDA")),
        "active_providers 不含 CUDA（静默回落 CPU？）：{providers:?}"
    );
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
