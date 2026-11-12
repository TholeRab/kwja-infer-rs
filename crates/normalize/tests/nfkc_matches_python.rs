//! 阶段 2 闸（normalize）：与 Python `normalize_text`（NFKC + TRANSLATION_TABLE）逐字符一致。
//! fixtures 由 阶段一/导出Rust侧阶段2基准.py 用 venv + 本仓 constants.py 生成。

use kwja_normalize::nfkc::normalize_text;

#[test]
fn nfkc_matches_python() {
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/normalize_cases.json"
    ))
    .unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert!(!cases.is_empty());
    for case in &cases {
        let input = case["input"].as_str().unwrap();
        let expected = case["expected"].as_str().unwrap();
        assert_eq!(normalize_text(input), expected, "input={input:?}");
    }
}
