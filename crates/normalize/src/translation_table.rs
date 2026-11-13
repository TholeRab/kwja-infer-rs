//! TRANSLATION_TABLE：3 对替换 + 68 个删除码点。
//!
//! 删除集合为 ASCII 控制符（0..32）、DEL 与 C1（127..160）、OGHAM 空格（5760）、LS/PS（8232、8233）。

/// 替换对：`"` 改作 `”`，`#` 改作 `＃`，`▁` 改作 `▂`。
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
