//! TRANSLATION_TABLE：3 对替换 + 68 个删除码点。
//!
//! 逐字对照 Python `代码/src/kwja/utils/constants.py:195-197`：
//! `str.maketrans('"#▁', "”＃▂", "".join(chr(i) for i in [*range(32), *range(127, 160), 5760, *range(8232, 8234)]))`

/// 替换对：`"` → `”`、`#` → `＃`、`▁` → `▂`。
pub const REPLACEMENTS: [(char, char); 3] = [('"', '”'), ('#', '＃'), ('▁', '▂')];

/// 该码点是否属于删除集合（控制字符 / DEL+C1 / OGHAM 空格 / LS / PS）。
#[must_use]
pub fn is_deleted(c: char) -> bool {
    let u = c as u32;
    u < 32 || (127..160).contains(&u) || u == 5760 || u == 8232 || u == 8233
}

/// 查表：返回替换后的字符；`None` 表示删除。
#[must_use]
pub fn translate(c: char) -> Option<char> {
    for (from, to) in REPLACEMENTS {
        if c == from {
            return Some(to);
        }
    }
    if is_deleted(c) {
        None
    } else {
        Some(c)
    }
}
