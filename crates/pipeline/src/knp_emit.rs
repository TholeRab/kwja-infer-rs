//! KNP 文本输出。
//!
//! 行构成只有三种：词素行、`* nD` 文节行、`+ nD <素性>` 基本句行。
//! 转义只对 surf/reading/lemma 三个字段：TAB 写作 \t，空格写作 \␣
//! （单反斜杠加 U+2423）。

use crate::morpheme_build::{FeatureValue, FlatMorpheme, OrderedFeats};

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\t' => out.push_str("\\t"),
            ' ' => out.push_str("\\\u{2423}"),
            other => out.push(other),
        }
    }
    out
}

/// 拼一条词素行。
pub fn morpheme_line(m: &FlatMorpheme) -> String {
    let mut parts: Vec<String> = vec![
        esc(&m.surf),
        esc(&m.reading),
        esc(&m.lemma),
        m.pos.clone(),
        m.pos_id.to_string(),
        m.subpos.clone(),
        m.subpos_id.to_string(),
        m.conjtype.clone(),
        m.conjtype_id.to_string(),
        m.conjform.clone(),
        m.conjform_id.to_string(),
    ];
    if !m.semantics.is_empty() {
        parts.push(m.semantics.clone());
    }
    // features 按插入序拼（参考实现不排序）；
    //   值里的 '>' 转成 '\>'。
    let mut feats = String::new();
    for (k, v) in m.features.iter() {
        match v {
            FeatureValue::True => feats.push_str(&format!("<{k}>")),
            FeatureValue::Text(t) => feats.push_str(&format!("<{k}:{}>", t.replace('>', "\\>"))),
        }
    }
    // 同形词 ALT 标签并进 features
    for a in &m.homographs {
        feats.push_str(&format!("<{a}>"));
    }
    if !feats.is_empty() {
        parts.push(feats);
    }
    parts.join(" ")
}

/// 一个基本句的全部素性串：按插入序拼，不排序（排序会与参考输出不一致）。
pub fn feature_string(feats: &OrderedFeats) -> String {
    let mut items: Vec<String> = Vec::new();
    for (k, v) in feats.iter() {
        match v {
            FeatureValue::True => items.push(k.clone()),
            FeatureValue::Text(t) => items.push(format!("{k}:{t}")),
        }
    }
    items.iter().map(|f| format!("<{f}>")).collect()
}

/// 拼基本句行（+ nD 行头与 rel/features 排布）。
///
/// `rels`：[(rel, target, sid, bp_index)]；sid 为 None 表示指外（无 id）。
pub fn bp_header(parent: i32, feats: &str, rels: &[(String, String, Option<String>, Option<usize>)], dep_type: Option<&str>) -> String {
    let mut buf: Vec<String> = Vec::new();
    for (t, tgt, sid, bidx) in rels {
        match (sid, bidx) {
            (None, _) => buf.push(format!("<rel type=\"{t}\" target=\"{tgt}\"/>")),
            (Some(sid), Some(bidx)) => {
                buf.push(format!("<rel type=\"{t}\" target=\"{tgt}\" sid=\"{sid}\" id=\"{bidx}\"/>"))
            }
            (Some(sid), None) => buf.push(format!("<rel type=\"{t}\" target=\"{tgt}\" sid=\"{sid}\"/>")),
        }
    }
    let dt = dep_type.unwrap_or("D");
    let mut s = format!("+ {}{dt}", if parent >= 0 { parent.to_string() } else { "-1".to_string() });
    // rel 与 features 之间不能加空格；没有 rel 时 features 前要空格。
    if !buf.is_empty() {
        s.push(' ');
        s.push_str(&buf.join(""));
        if !feats.is_empty() {
            s.push_str(feats);
        }
    } else if !feats.is_empty() {
        s.push(' ');
        s.push_str(feats);
    }
    s
}
