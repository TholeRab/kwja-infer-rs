//! char_stage 的纯函数单测。
use kwja_pipeline::char_stage::{parse_raw_text, split_document, to_orig_doc_id};

#[test]
fn parse_raw_text_reads_did_and_strips() {
    // Python: '# D-ID:c-CH-0001'.split()[1].split(':')[1] == 'c-CH-0001'
    let (doc_id, text) = parse_raw_text("# D-ID:c-CH-0001\n　さっきまで、水瀬が使っていたもの。\n");
    assert_eq!(doc_id, "c-CH-0001");
    // strip 只去两端空白（含全角空格），中间不动
    assert_eq!(text, "さっきまで、水瀬が使っていたもの。");
    // 无 D-ID 行时 doc_id 留空，由调用方补齐
    let (doc_id, text) = parse_raw_text("一行目\n二行目");
    assert_eq!(doc_id, "");
    assert_eq!(text, "一行目\n二行目");
}

#[test]
fn parse_raw_text_takes_second_colon_field_like_python() {
    // '# D-ID:a:b'.split()[1].split(':')[1] == 'a'
    let (doc_id, _) = parse_raw_text("# D-ID:a:b\n本文。");
    assert_eq!(doc_id, "a");
}

#[test]
fn to_orig_doc_id_strips_window_suffix_only() {
    assert_eq!(to_orig_doc_id("c-00-s12i3"), "c-00");
    assert_eq!(to_orig_doc_id("run0-s1i0"), "run0");
    assert_eq!(to_orig_doc_id("c-0"), "c-0");
}

#[test]
fn split_document_balances_brackets_and_merges_period_runs() {
    // 以下期望值与 Python 参考实现一致
    assert_eq!(split_document("「ああ」。いい。"), vec!["「ああ」。", "いい。"]);
    // 括号跨句末符未闭合时，候选往后接
    assert_eq!(split_document("「ああ。いい。」"), vec!["「ああ。いい。」"]);
    // 连续句末符算一个（正则的 + 号）
    assert_eq!(split_document("すごい！！本当？"), vec!["すごい！！", "本当？"]);
    // 行尾仍不闭合时，剩余小段再按句末符切
    assert_eq!(split_document("「ああ。いい"), vec!["「ああ。", "いい"]);
    assert_eq!(split_document("（あ。い。"), vec!["（あ。", "い。"]);
}
