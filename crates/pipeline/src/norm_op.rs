//! get_normalized_surf 的逐分支复刻，行为与 Python 参考实现一致。
//!
//! char 阶段的调用口径是 strict=false：模型预测出「非法组合」时
//! 不报错，按宽松分支原样保留。P/E 依赖已累积 norm 的最后一个字，
//! 所以必须按顺序串行处理，不能对每个字符独立查表。

use crate::char_tables::{
    CHOON_SET, HATSUON_SET, LOWER2UPPER, PROLONGED_MAP, PROLONGED_MAP_FOR_EROW, VOICED2VOICELESS,
};

fn in_set(set: &[char], c: char) -> bool {
    set.contains(&c)
}

fn map_lookup(table: &[(char, char)], c: char) -> Option<char> {
    table.iter().find(|(from, _)| *from == c).map(|(_, to)| *to)
}

/// 与 Python 同名函数一致：`surf` 与 `tags` 逐字符对齐。
/// `strict=true` 的报错路径也照抄（char 阶段不走，但行为要能对上）。
///
/// # Errors
/// 长度不一致、`strict=true` 下的非法组合、或未知标签（对应 Python 的
/// `NotImplementedError`）时返回 `Err`。
pub fn get_normalized_surf(surf: &str, tags: &[&str], strict: bool) -> Result<String, String> {
    if surf.chars().count() != tags.len() {
        return Err("Surface and normalization tags must have the same length".to_string());
    }
    let mut norm = String::new();
    for (i, (c, tag)) in surf.chars().zip(tags.iter()).enumerate() {
        match *tag {
            "K" => norm.push(c),
            "V" => {
                if strict && i != 0 {
                    return Err(format!("not an initial kana {c} in {surf}"));
                }
                if let Some(t) = map_lookup(&VOICED2VOICELESS, c) {
                    norm.push(t);
                } else {
                    if strict {
                        return Err(format!("not a voiced kana {c} in {surf}"));
                    }
                    norm.push(c);
                }
            }
            "D" => {
                if strict
                    && !in_set(&CHOON_SET, c)
                    && !in_set(&HATSUON_SET, c)
                    && map_lookup(&LOWER2UPPER, c).is_none()
                {
                    return Err(format!("not a removable kana {c} in {surf}"));
                }
                // 删除：不追加任何字符
            }
            "S" => {
                if let Some(t) = map_lookup(&LOWER2UPPER, c) {
                    norm.push(t);
                } else {
                    if strict {
                        return Err(format!("not a small kana {c} in {surf}"));
                    }
                    norm.push(c);
                }
            }
            "P" => {
                // NOTE: 规范的 ops 里 P 必须跟在 K 后，但这里与 Python 一样不查该约束
                if norm.is_empty() {
                    if strict {
                        return Err(format!("no preceding kana for {c} in {surf}"));
                    }
                    norm.push(c);
                } else if !in_set(&CHOON_SET, c) {
                    if strict {
                        return Err(format!("not a prolonged sign for {c} in {surf}"));
                    }
                    norm.push(c);
                } else {
                    let p = norm.chars().last().expect("norm 非空");
                    if let Some(t) = map_lookup(&PROLONGED_MAP, p) {
                        norm.push(t);
                    } else {
                        if strict {
                            return Err(format!("not a valid preceding kana {p} in {surf}"));
                        }
                        norm.push(c);
                    }
                }
            }
            "E" => {
                if norm.is_empty() {
                    if strict {
                        return Err(format!("no preceding kana for {c} in {surf}"));
                    }
                    norm.push(c);
                } else if !in_set(&CHOON_SET, c) {
                    if strict {
                        return Err(format!("not a prolonged sign for {c} in {surf}"));
                    }
                    norm.push(c);
                } else {
                    let p = norm.chars().last().expect("norm 非空");
                    if let Some(t) = map_lookup(&PROLONGED_MAP_FOR_EROW, p) {
                        norm.push(t);
                    } else {
                        if strict {
                            return Err(format!("not a valid preceding kana {p} in {surf}"));
                        }
                        norm.push(c);
                    }
                }
            }
            other => return Err(format!("unknown word norm op tag {other}")),
        }
    }
    Ok(norm)
}
