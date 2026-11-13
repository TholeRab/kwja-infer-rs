//! char 三头标签到句子、词素、Juman++ 行的转换。
//!
//! 必须保持的三处细节：
//! 1. 半角空格强制断词，且空格自己单独成一个词素；
//! 2. 词素编号逐句重置——本实现按「每句一个独立 Vec」天然满足；
//! 3. get_normalized_surf 的切片按全文档游标减去当前词长度取窗，不是每句重数。

use crate::norm_op::get_normalized_surf;

pub const SENT_SEGMENTATION_TAGS: [&str; 2] = ["B", "I"];
pub const WORD_SEGMENTATION_TAGS: [&str; 2] = ["B", "I"];
pub const WORD_NORM_OP_TAGS: [&str; 6] = ["K", "D", "V", "S", "P", "E"];

/// argmax 语义：并列时取第一个最大值的下标。
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max {
            max = v;
            best = i;
        }
    }
    best
}

/// 按非 special 位下标把三路预测挑成标签串。
pub fn convert_char_predictions_into_tags(
    sent_segmentation_predictions: &[usize],
    word_segmentation_predictions: &[usize],
    word_norm_op_predictions: &[usize],
    indices: &[usize],
) -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    let sent = indices
        .iter()
        .map(|&i| SENT_SEGMENTATION_TAGS[sent_segmentation_predictions[i]])
        .collect();
    let word = indices
        .iter()
        .map(|&i| WORD_SEGMENTATION_TAGS[word_segmentation_predictions[i]])
        .collect();
    let norm = indices
        .iter()
        .map(|&i| WORD_NORM_OP_TAGS[word_norm_op_predictions[i]])
        .collect();
    (sent, word, norm)
}

/// char 阶段造出的词素：11 个 Juman++ 字段里除 surf/lemma 外全是常量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Morpheme {
    pub surf: String,
    pub reading: String,
    pub lemma: String,
    pub pos: &'static str,
    pub pos_id: u32,
    pub subpos: &'static str,
    pub subpos_id: u32,
    pub conjtype: &'static str,
    pub conjtype_id: u32,
    pub conjform: &'static str,
    pub conjform_id: u32,
    /// char 阶段恒为空串；`morpheme_line` 里按 Python 的判空语义保留分支。
    pub semantics: String,
}

/// lemma = norm or surf（norm 为空串时回落 surf）。
#[must_use]
pub fn build_morpheme(surf: &str, norm: &str) -> Morpheme {
    Morpheme {
        surf: surf.to_string(),
        reading: "_".to_string(),
        lemma: if norm.is_empty() { surf.to_string() } else { norm.to_string() },
        pos: "未定義語",
        pos_id: 15,
        subpos: "その他",
        subpos_id: 1,
        conjtype: "*",
        conjtype_id: 0,
        conjform: "*",
        conjform_id: 0,
        semantics: String::new(),
    }
}

/// 按句界标签把原文切成句子文本。
#[must_use]
pub fn set_sentences_plain(text: &str, sent_segmentation_tags: &[&str]) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut surf = String::new();
    for (c, tag) in text.chars().zip(sent_segmentation_tags.iter()) {
        if *tag == "B" && !surf.is_empty() {
            sentences.push(std::mem::take(&mut surf));
        }
        surf.push(c);
    }
    if !surf.is_empty() {
        sentences.push(surf);
    }
    sentences
}

/// 按分词标签把每句切成词素；游标跨句连续。
///
/// # Panics
/// `strict=false` 下 `get_normalized_surf` 不应失败；失败视为程序错误。
pub fn set_morphemes_plain(
    sentence_texts: &[String],
    word_segmentation_tags: &[&str],
    word_norm_op_tags: &[&str],
) -> Vec<Vec<Morpheme>> {
    let mut out: Vec<Vec<Morpheme>> = Vec::new();
    let mut cursor = 0usize;
    for sentence_text in sentence_texts {
        let mut morphemes: Vec<Morpheme> = Vec::new();
        let mut surf = String::new();
        for ch in sentence_text.chars() {
            // 半角空格也强制断词
            if (word_segmentation_tags[cursor] == "B" || ch == ' ') && !surf.is_empty() {
                let start = cursor - surf.chars().count();
                let norm = get_normalized_surf(&surf, &word_norm_op_tags[start..cursor], false)
                    .expect("strict=false 下不应失败");
                morphemes.push(build_morpheme(&surf, &norm));
                surf.clear();
            }
            if ch == ' ' {
                // 空格自己单独成一个词素
                morphemes.push(build_morpheme(" ", " "));
            } else {
                surf.push(ch);
            }
            cursor += 1;
        }
        if !surf.is_empty() {
            let start = cursor - surf.chars().count();
            let norm = get_normalized_surf(&surf, &word_norm_op_tags[start..cursor], false)
                .expect("strict=false 下不应失败");
            morphemes.push(build_morpheme(&surf, &norm));
        }
        out.push(morphemes);
    }
    out
}

/// 转义表（char 阶段只涉及制表符与空格）。
fn esc(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\t' => "\\t".to_string(),
            ' ' => "\\␣".to_string(),
            _ => c.to_string(),
        })
        .collect()
}

/// 一条词素行。
#[must_use]
pub fn morpheme_line(m: &Morpheme) -> String {
    let mut parts = vec![
        esc(&m.surf),
        esc(&m.reading),
        esc(&m.lemma),
        m.pos.to_string(),
        m.pos_id.to_string(),
        m.subpos.to_string(),
        m.subpos_id.to_string(),
        m.conjtype.to_string(),
        m.conjtype_id.to_string(),
        m.conjform.to_string(),
        m.conjform_id.to_string(),
    ];
    if !m.semantics.is_empty() {
        parts.push(format!("\"{}\"", m.semantics));
    }
    // features 恒空（char 阶段造的词素没有素性）
    parts.join(" ")
}

/// 一句的 Juman++ 文本。
#[must_use]
pub fn sentence_jumanpp(comment: &str, morphemes: &[Morpheme]) -> String {
    let mut ret = String::new();
    if !comment.is_empty() {
        ret.push_str(comment);
        ret.push('\n');
    }
    for m in morphemes {
        ret.push_str(&morpheme_line(m));
        ret.push('\n');
    }
    ret.push_str("EOS\n");
    ret
}
