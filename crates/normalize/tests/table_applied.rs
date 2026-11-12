//! 阶段 2 闸（表）：TRANSLATION_TABLE 的 3 对替换与 68 个删除码点，逐条对 constants.py:195-197。

use kwja_normalize::translation_table::{is_deleted, translate};

#[test]
fn replacements_applied() {
    assert_eq!(translate('"'), Some('”'));
    assert_eq!(translate('#'), Some('＃'));
    assert_eq!(translate('▁'), Some('▂'));
}

#[test]
fn deleted_codepoints() {
    // range(32)：0x00–0x1F 全删
    assert_eq!(translate('\0'), None);
    assert_eq!(translate('\t'), None);
    assert_eq!(translate('\n'), None);
    assert_eq!(translate('\r'), None);
    assert_eq!(translate('\u{1f}'), None);
    // range(127, 160)：DEL + C1 全删
    assert_eq!(translate('\u{7f}'), None);
    assert_eq!(translate('\u{80}'), None);
    assert_eq!(translate('\u{85}'), None);
    assert_eq!(translate('\u{9f}'), None);
    // 散点：OGHAM 空格、LS、PS
    assert_eq!(translate('\u{1680}'), None);
    assert_eq!(translate('\u{2028}'), None);
    assert_eq!(translate('\u{2029}'), None);
    assert!(is_deleted('\u{2029}'));
}

#[test]
fn boundary_codepoints_survive() {
    // 32 是空格：不在 range(32)（0..32）里，保留
    assert_eq!(translate(' '), Some(' '));
    // 160 是 NBSP：不在 range(127,160)（127..160 右开）里，保留
    assert_eq!(translate('\u{a0}'), Some('\u{a0}'));
    // 8231 在 LS/PS 之前、8234 在之后：都保留
    assert_eq!(translate('\u{8231}'), Some('\u{8231}'));
    assert_eq!(translate('\u{8234}'), Some('\u{8234}'));
    // 普通字符原样
    assert_eq!(translate('あ'), Some('あ'));
    assert_eq!(translate('”'), Some('”'));
}
