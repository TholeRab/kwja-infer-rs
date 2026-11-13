//! normalize_text（NFKC 加 TRANSLATION_TABLE）与 Python 参考实现逐字符一致。
//! fixtures：tests/fixtures/normalize_cases.json。

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
