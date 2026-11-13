//! jumanpp 解析单测。
use kwja_pipeline::jumanpp::{parse_sentence, sid_of_comment, split_sentences, surf_of_line};

#[test]
fn split_sentences_keeps_eos_and_drops_blank_lines() {
    let text = "# S-ID:a-1 kwja:2.6.0\nりんご _ りんご 名詞 6 普通名詞 1 * 0 * 0\nEOS\n\n# S-ID:a-2 kwja:2.6.0\nみかん _ みかん 名詞 6 普通名詞 1 * 0 * 0\nEOS\n";
    let sentences = split_sentences(text);
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].ends_with("EOS\n"));
    assert!(sentences[1].starts_with("# S-ID:a-2"));
}

#[test]
fn parse_sentence_reads_sid_and_surfs() {
    let sent = "# S-ID:c-0-1 kwja:2.6.0\nさっき _ さっき 未定義語 15 その他 1 * 0 * 0\n、 _ 、 未定義語 15 その他 1 * 0 * 0\nEOS\n";
    let parsed = parse_sentence(sent);
    assert_eq!(parsed.sid, "c-0-1");
    assert_eq!(parsed.morphemes.len(), 2);
    assert_eq!(parsed.morphemes[0].surf, "さっき");
    assert_eq!(parsed.morphemes[1].surf, "、");
}

#[test]
fn surf_of_line_handles_leading_space_surf() {
    assert_eq!(surf_of_line("␣ _ ␣ 未定義語 15 その他 1 * 0 * 0").unwrap(), "␣");
    // surf 为半角空格时，行首是空格：` [^ ]*` 分支
    assert_eq!(surf_of_line("  _   未定義語 15 その他 1 * 0 * 0").unwrap(), " ");
}

#[test]
fn sid_of_comment_matches_python() {
    assert_eq!(sid_of_comment("# S-ID:c-58-214 kwja:2.6.0"), "c-58-214");
    assert_eq!(sid_of_comment("# 其他注释"), "");
    assert_eq!(sid_of_comment(""), "");
}
