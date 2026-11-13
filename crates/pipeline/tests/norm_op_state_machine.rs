//! get_normalized_surf 行为测试：strict=false（char 阶段口径）下逐用例与
//! Python 参考实现全等；strict=true 的合法/报错路径抽查。

use kwja_pipeline::norm_op::get_normalized_surf;

#[test]
fn norm_op_loose_branch_matches_python() {
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/norm_op_cases.json"
    ))
    .unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert!(!cases.is_empty());
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let surf = case["surf"].as_str().unwrap();
        let tags: Vec<&str> = case["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let expected = case["expected"].as_str().unwrap();
        let got = get_normalized_surf(surf, &tags, false)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, expected, "case={name}");
    }
}

#[test]
fn strict_branch_matches_python() {
    // 合法组合：strict 与 loose 同结果
    assert_eq!(get_normalized_surf("かー", &["K", "P"], true).unwrap(), "かあ");
    assert_eq!(get_normalized_surf("けー", &["K", "E"], true).unwrap(), "けえ");
    assert_eq!(get_normalized_surf("が", &["V"], true).unwrap(), "か");
    assert_eq!(get_normalized_surf("ぁ", &["S"], true).unwrap(), "あ");
    assert_eq!(get_normalized_surf("あー", &["K", "D"], true).unwrap(), "あ");
    // あ 不在 PROLONGED_MAP：strict 报错，宽松保留
    assert!(get_normalized_surf("あー", &["K", "P"], true).is_err());
    // 非法组合：strict 必须 Err（对应 Python raise ValueError）
    assert!(get_normalized_surf("あが", &["K", "V"], true).is_err()); // V 不在词首
    assert!(get_normalized_surf("あ", &["V"], true).is_err()); // 非浊音
    assert!(get_normalized_surf("い", &["D"], true).is_err()); // 非可删字
    assert!(get_normalized_surf("い", &["S"], true).is_err()); // 非小字
    assert!(get_normalized_surf("ー", &["P"], true).is_err()); // 无前字
    assert!(get_normalized_surf("い", &["P"], true).is_err()); // 非长音符
    assert!(get_normalized_surf("んー", &["K", "P"], true).is_err()); // 前字不在映射
    assert!(get_normalized_surf("いー", &["K", "E"], true).is_err()); // E 同
    // 长度不一致：ValueError
    assert!(get_normalized_surf("あ", &["K", "K"], false).is_err());
    // 未知标签：NotImplementedError
    assert!(get_normalized_surf("あ", &["X"], false).is_err());
}
