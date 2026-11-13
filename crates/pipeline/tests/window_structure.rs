//! stride=2 分窗结构测试：窗数与每窗目标句下标和 Python 参考实现全等。
//!
//! 基线 fixtures 在 tests/fixtures/：输入是 char 阶段已逐字节验证的产出
//! （即 word 阶段的真实输入），超参取 max_seq_length=256、
//! document_split_stride=2、max_token_length=246。
//! 本测试读基线，用本仓实现重算分窗结构并逐字节比对。
//!
//! doc_id 前缀 Python 侧默认取时间戳，比对无意义，
//! 两侧都只保留 -s{stride}i{idx} 后缀（单窗文档记作 DOC）。

use std::path::PathBuf;

use kwja_pipeline::word_stage::{parse_jumanpp_document, split_flat_word_document};
use kwja_tokenizer::word_side::WordTokenizer;

/// word 阶段配置里的 special_tokens：
/// additional_special_tokens 是 8 个（含 " "，决定 tokenizer 怎么吃空格）；
/// 数据集内部用的 special_tokens 滤掉 " " 剩 7 个。
const ADDITIONAL_SPECIAL_TOKENS: [&str; 8] = [
    "[著者]",
    "[読者]",
    "[不特定:人]",
    "[不特定:物]",
    "[NULL]",
    "[NA]",
    "[ROOT]",
    " ",
];
const SPECIAL_TOKENS: [&str; 7] = [
    "[著者]",
    "[読者]",
    "[不特定:人]",
    "[不特定:物]",
    "[NULL]",
    "[NA]",
    "[ROOT]",
];

/// 仓内素材路径：从 crate 清单目录上溯找到仓根，再拼相对路径。
fn repo_path(rel: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn word_tokenizer() -> WordTokenizer {
    let snapshots = repo_path(
        "assets/tokenizers/models--ku-nlp--deberta-v2-base-japanese/snapshots",
    );
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .expect("word tokenizer snapshots 目录不存在")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    hits.sort();
    assert_eq!(hits.len(), 1, "word tokenizer 快照不唯一：{hits:?}");
    WordTokenizer::load(
        hits.remove(0).join("tokenizer.json"),
        &ADDITIONAL_SPECIAL_TOKENS,
        &SPECIAL_TOKENS,
        256,
    )
    .expect("WordTokenizer 加载失败")
}

/// doc_id 里剥掉前缀，只留 `-s{stride}i{idx}` 后缀（与 Python 基线的归一化一致）。
fn sub_suffix(doc_id: &str) -> Option<(String, String)> {
    let re = regex::Regex::new(r"-s(\d+)i(\d+)$").expect("后缀正则编译失败");
    re.captures(doc_id)
        .map(|c| (c[1].to_string(), c[2].to_string()))
}

/// 重算分窗结构，产出与基线同格式的文本。
fn dump_windows(
    juman_text: &str,
    doc_id: &str,
    max_token_length: usize,
    stride: i64,
    tk: &WordTokenizer,
) -> String {
    let document = parse_jumanpp_document(juman_text, doc_id);
    let windows =
        split_flat_word_document(&document, max_token_length, tk, stride).expect("分窗失败");
    let total_tokens: usize = windows
        .iter()
        .map(|w| tk.surf_token_len(&w.surfs()).expect("计 token 失败"))
        .sum();
    let mut lines = vec![
        "# max_seq_length=256".to_string(),
        format!("# max_token_length={max_token_length}"),
        format!("# stride={stride}"),
        format!("# windows={}", windows.len()),
        format!("# morphemes_tokens={total_tokens}"),
    ];
    for window in &windows {
        let flags = window.target_sentence_flags();
        let idxs: Vec<String> = flags
            .iter()
            .enumerate()
            .filter(|(_, flag)| **flag)
            .map(|(i, _)| i.to_string())
            .collect();
        let label = match sub_suffix(&window.doc_id) {
            Some((s, i)) => format!("s{s}i{i}"),
            None => "DOC".to_string(),
        };
        lines.push(format!(
            "{label}\t{}\t{}",
            window.sentences.len(),
            idxs.join(",")
        ));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// 失败时打印长度、行号与首个差异行，便于定位是哪一层开始偏的。
fn assert_dump_equal(produced: &str, expected: &str, name: &str) {
    if produced == expected {
        return;
    }
    let produced_lines: Vec<&str> = produced.lines().collect();
    let expected_lines: Vec<&str> = expected.lines().collect();
    let mut first_diff = String::from("（行数不同）");
    if produced_lines.len() == expected_lines.len() {
        for (i, (p, e)) in produced_lines.iter().zip(expected_lines.iter()).enumerate() {
            if p != e {
                first_diff = format!("第 {} 行\n  Rust  ：{p:?}\n  Python：{e:?}", i + 1);
                break;
            }
        }
    }
    panic!(
        "分窗结构与 Python 基线不一致：{name}\nRust {} 行 / Python {} 行\n{}",
        produced_lines.len(),
        expected_lines.len(),
        first_diff
    );
}

fn check(name: &str) {
    let fixture = std::fs::read_to_string(fixtures_dir().join(format!("word_windows_{name}.txt")))
        .expect("word 分窗基线缺失");
    // 头部元信息以 Python 侧为准，测试不做任何假设
    let mut max_token_length: Option<usize> = None;
    let mut stride: Option<i64> = None;
    for line in fixture.lines().take(5) {
        if let Some(v) = line.strip_prefix("# max_token_length=") {
            max_token_length = Some(v.parse().expect("max_token_length 应为非负整数"));
        }
        if let Some(v) = line.strip_prefix("# stride=") {
            stride = Some(v.parse().expect("stride 应为整数"));
        }
    }
    let max_token_length = max_token_length.expect("基线头部缺 max_token_length");
    let stride = stride.expect("基线头部缺 stride");

    let juman_text =
        std::fs::read_to_string(fixtures_dir().join(format!("char_stage_{name}.txt")))
            .expect("char 产出基线缺失");
    let produced = dump_windows(&juman_text, "doc", max_token_length, stride, &word_tokenizer());
    assert_dump_equal(&produced, &fixture, name);
}

#[test]
fn chapter1_window_structure_matches_python() {
    check("ch1");
}

#[test]
fn whole_book_window_structure_matches_python() {
    check("all");
}
