//! 阶段 2 闸（tokenizer）：char / word 两侧的 input_ids・attention_mask 与 Python 完全相同。
//! fixtures 由 阶段一/导出Rust侧阶段2基准.py 用 venv transformers 生成
//! （char: padding=max_length/max_length=512；word: is_split_into_words + special 7 + pad 到 256）。

use kwja_tokenizer::char_side::CharTokenizer;
use kwja_tokenizer::word_side::WordTokenizer;
use std::path::PathBuf;

/// 仓内素材路径：从 crate 清单目录上溯找到仓根，再拼相对路径。
/// ★ 起素材一律在本仓 assets/ 下，不再引用 参考实现仓。
fn repo_path(rel: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("仓内找不到 {}", rel))
}

fn tokenizer_json(hub_dir_name: &str) -> PathBuf {
    let snapshots = repo_path("assets/tokenizers")
        .join(hub_dir_name)
        .join("snapshots");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .expect("snapshots 目录不存在")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("tokenizer.json").exists())
        .collect();
    assert_eq!(hits.len(), 1, "{hub_dir_name}: snapshots 不唯一");
    hits.remove(0).join("tokenizer.json")
}

fn vocab_txt(hub_dir_name: &str) -> PathBuf {
    let snapshots = repo_path("assets/tokenizers")
        .join(hub_dir_name)
        .join("snapshots");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .expect("snapshots 目录不存在")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("vocab.txt").exists())
        .collect();
    assert_eq!(hits.len(), 1, "{hub_dir_name}: snapshots 不唯一");
    hits.remove(0).join("vocab.txt")
}

fn load_cases(file: &str) -> Vec<serde_json::Value> {
    let json = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/").to_string() + file,
    )
    .unwrap();
    serde_json::from_str(&json).unwrap()
}

fn ids(case: &serde_json::Value, key: &str) -> Vec<i64> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

#[test]
fn char_ids_match_python() {
    let cases = load_cases("char_cases.json");
    assert!(!cases.is_empty());
    // ★char 快照里没有 tokenizer.json（只有 vocab.txt；方案 §二 写的位置实测不存在），
    // Python 侧拿到的是慢分词器 BertJapaneseTokenizer——Rust 侧按其源码逐 id 复刻。
    let tok = CharTokenizer::load(
        vocab_txt("models--ku-nlp--deberta-v2-base-japanese-char-wwm"),
        &[" "],
        512,
    )
    .unwrap();
    for case in &cases {
        let text = case["text"].as_str().unwrap();
        let got = tok.encode(text).unwrap();
        assert_eq!(got.input_ids, ids(case, "ids"), "char ids 不一致：{text:?}");
        assert_eq!(
            got.attention_mask,
            ids(case, "mask"),
            "char mask 不一致：{text:?}"
        );
    }
}

#[test]
fn word_ids_match_python() {
    let cases = load_cases("word_cases.json");
    assert!(!cases.is_empty());
    let tok = WordTokenizer::load(
        tokenizer_json("models--ku-nlp--deberta-v2-base-japanese"),
        &["[著者]", "[読者]", "[不特定:人]", "[不特定:物]", "[NULL]", "[NA]", "[ROOT]", " "],
        &["[著者]", "[読者]", "[不特定:人]", "[不特定:物]", "[NULL]", "[NA]", "[ROOT]"],
        256,
    )
    .unwrap();
    for case in &cases {
        let surfs: Vec<String> = case["surfs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let got = tok.encode_words(&surfs).unwrap();
        assert_eq!(
            got.input_ids,
            ids(case, "ids"),
            "word ids 不一致：{surfs:?}"
        );
        assert_eq!(
            got.attention_mask,
            ids(case, "mask"),
            "word mask 不一致：{surfs:?}"
        );
    }
}
