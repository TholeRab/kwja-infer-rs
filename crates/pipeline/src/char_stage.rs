//! char 阶段（阶段 5）：RAW 章节文本 → Juman++ 文本，与 Python 扁平链逐字节对齐。
//!
//! 数据流（对照 Python 侧）：
//! 1. `# D-ID:` 解析 ＋ 原文拼接（`flat_utils.py:22-40`）；
//! 2. doc_id 补齐：`{prefix}-{按文档数零填充的序号}`（`flat_utils.py:43-50`）；
//! 3. 正则分句（`flat_senter.py:20-58`，逐字照抄 rhoknp RegexSenter）；
//! 4. 按 token 数贪心切窗，窗间不重叠（`flat_utils.py:79-120`，
//!    `max_token_length = max_seq_length - len(extra_special_tokens) - 2`）；
//! 5. 每窗：tokenize（[CLS]+ids+[SEP]+pad）→ char encoder → 三头逐位 argmax →
//!    非 special 位过滤 → 标签 → 分句/分词/归一化 → Juman++ 行
//!    （`char_module_writer.py:46-85`）。
//!
//! sid 口径：`# S-ID:{orig_doc_id}-{sid} kwja:{version}`，sid 从 1 起、
//! 跨窗连续、orig_doc_id 变化时重置（`char_module_writer.py:71-84`）。

use std::path::Path;
use std::sync::OnceLock;

use kwja_encoder::{CharEncoderInput, Encoder};
use kwja_tokenizer::char_side::CharTokenizer;
use regex::Regex;

use crate::char_tags::{
    convert_char_predictions_into_tags, sentence_jumanpp, set_morphemes_plain, set_sentences_plain,
};

/// 与 venv 里的 kwja 2.6.0 一致（基线脚本断言过）。
pub const KWJA_VERSION: &str = "2.6.0";

/// char 阶段的 tokenizer 附加 special：与 CLI 配置一致（`additional_special_tokens=[" "]`）。
const CHAR_ADDITIONAL_SPECIAL_TOKENS: [&str; 1] = [" "];

pub struct CharStage {
    tokenizer: CharTokenizer,
    max_seq_length: usize,
    extra_special_tokens: usize,
}

impl CharStage {
    pub fn new(vocab_txt: impl AsRef<Path>, max_seq_length: usize) -> Result<Self, String> {
        let tokenizer = CharTokenizer::load(
            vocab_txt,
            &CHAR_ADDITIONAL_SPECIAL_TOKENS,
            max_seq_length,
        )?;
        // max_token_length = max_seq_length - len(extra_special_tokens) - 2
        // （char_inference.py:59；extra_special_tokens = [' ']，长度 1）
        let extra_special_tokens = CHAR_ADDITIONAL_SPECIAL_TOKENS.len();
        Ok(Self {
            tokenizer,
            max_seq_length,
            extra_special_tokens,
        })
    }

    /// 跑完整个 char 阶段，返回 Juman++ 文本。
    ///
    /// `texts`：每章一段**已归一化**（NFKC＋转义表）的原文——CLI 流程里
    /// normalize_text 发生在写输入文件时（`cli.py:276`），归一化不属于本模块。
    /// `doc_id_prefix`：doc_id 为空时的前缀；Python 默认取时间戳，Rust 侧
    /// 要求显式给出（None 时退回 "c"），保证产出可复现、可比对。
    pub fn run(
        &self,
        texts: &[String],
        doc_id_prefix: Option<&str>,
        encoder: &mut dyn Encoder,
    ) -> Result<String, String> {
        // 1-2. 解析 D-ID 与 doc_id 补齐
        let mut documents: Vec<(String, String)> =
            texts.iter().map(|t| parse_raw_text(t)).collect();
        let prefix = doc_id_prefix.unwrap_or("c");
        let width = documents.len().to_string().len();
        for (idx, (doc_id, _)) in documents.iter_mut().enumerate() {
            if doc_id.is_empty() {
                *doc_id = format!("{prefix}-{idx:0width$}");
            }
        }

        // 3-4. 正则分句 ＋ 按 token 数切窗
        let max_token_length = self.max_seq_length - self.extra_special_tokens - 2;
        let mut windows: Vec<(String, String)> = Vec::new();
        for (doc_id, text) in &documents {
            let sentences = split_document(text);
            windows.extend(split_flat_document(
                &sentences,
                doc_id,
                text,
                max_token_length,
                &self.tokenizer,
            ));
        }

        // 5. 逐窗前向 ＋ 拼 Juman++
        let special_ids = self.tokenizer.writer_special_ids();
        let mut output = String::new();
        let mut prev_doc_id = String::new();
        let mut prev_sid = 0u64;
        for (doc_id, text) in &windows {
            let encoding = self.tokenizer.encode(text)?;
            // 超长窗照 Python 丢弃（char_inference.py:76-78 的 warning + continue）
            if encoding.input_ids.len() > self.max_seq_length {
                continue;
            }
            let seq = encoding.input_ids.len();
            let logits = encoder
                .run_char(&CharEncoderInput {
                    input_ids: encoding.input_ids.clone(),
                    attention_mask: encoding.attention_mask.clone(),
                    seq,
                })
                .map_err(|e| format!("char 前向失败：{e}"))?;
            let sent_pred: Vec<usize> = (0..seq)
                .map(|i| {
                    crate::char_tags::argmax(
                        logits.sent_segmentation.slice(ndarray::s![0, i, ..])
                            .as_slice()
                            .unwrap_or(&[]),
                    )
                })
                .collect();
            let word_pred: Vec<usize> = (0..seq)
                .map(|i| {
                    crate::char_tags::argmax(
                        logits.word_segmentation.slice(ndarray::s![0, i, ..])
                            .as_slice()
                            .unwrap_or(&[]),
                    )
                })
                .collect();
            let norm_pred: Vec<usize> = (0..seq)
                .map(|i| {
                    crate::char_tags::argmax(
                        logits.word_norm_op.slice(ndarray::s![0, i, ..])
                            .as_slice()
                            .unwrap_or(&[]),
                    )
                })
                .collect();
            let indices: Vec<usize> = (0..seq)
                .filter(|&i| !special_ids.contains(&(encoding.input_ids[i] as u32)))
                .collect();
            let (sent_tags, word_tags, norm_tags) =
                convert_char_predictions_into_tags(&sent_pred, &word_pred, &norm_pred, &indices);
            // set_sentences_plain 的 zip(strict=True) 不变量：标签数 == 文本字符数。
            // 错位说明 tokenizer 归一化改了字符数，Python 在这里抛 ValueError。
            if text.chars().count() != sent_tags.len() {
                return Err(format!(
                    "分句标签数（{}）与文本字符数（{}）不一致：{text:?}",
                    sent_tags.len(),
                    text.chars().count()
                ));
            }
            let sentence_texts = set_sentences_plain(text, &sent_tags);
            let per_sentence_morphemes =
                set_morphemes_plain(&sentence_texts, &word_tags, &norm_tags);

            let orig_doc_id = to_orig_doc_id(doc_id);
            if orig_doc_id != prev_doc_id {
                prev_doc_id = orig_doc_id.clone();
                prev_sid = 1; // 1-origin
            }
            for morphemes in &per_sentence_morphemes {
                let comment = format!("# S-ID:{orig_doc_id}-{prev_sid} kwja:{KWJA_VERSION}");
                output.push_str(&sentence_jumanpp(&comment, morphemes));
                prev_sid += 1;
            }
        }
        Ok(output)
    }
}

/// 解析一段原文：`# D-ID:` 行取 doc_id，其余行拼回原文并 strip
/// （`flat_utils.py:22-40`；strip 必须照抄，模型标签按 strip 后的长度给）。
/// Python 取 id 的写法是 `line.split()[1].split(":")[1]`，这里照抄；
/// 格式残缺行 Python 会抛 IndexError，这里按普通原文行处理（语料里不出现）。
fn parse_raw_text(text: &str) -> (String, String) {
    let mut raw_text = String::new();
    let mut doc_id = String::new();
    for line in text.split('\n') {
        let mut handled = false;
        if line.starts_with("# D-ID:") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if let Some(id) = fields.get(1).and_then(|f| f.split(':').nth(1)) {
                doc_id = id.to_string();
                handled = true;
            }
        }
        if !handled {
            raw_text.push_str(line);
            raw_text.push('\n');
        }
    }
    (doc_id, raw_text.trim().to_string())
}

fn period_pattern() -> &'static Regex {
    static PAT: OnceLock<Regex> = OnceLock::new();
    PAT.get_or_init(|| Regex::new(r"[。．？！♪☆★…?!]+").expect("句末符正则编译失败"))
}

/// 按句末符把一行切成候选段（`flat_senter.py:25-34`）。
fn split_text_by_period(s: &str) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut start = 0usize;
    for mat in period_pattern().find_iter(s) {
        let end = mat.end();
        segments.push(s[start..end].to_string());
        start = end;
    }
    if start < s.len() {
        segments.push(s[start..].to_string());
    }
    segments.iter().map(|seg| seg.trim().to_string()).collect()
}

/// 正则分句：先按行切、再按句末符切、再用三种括号配平把候选合回去
/// （`flat_senter.py:20-58`，与 rhoknp RegexSenter 同算法）。
pub fn split_document(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut sentences: Vec<String> = Vec::new();
    for line in text.split('\n') {
        let mut candidates: std::collections::VecDeque<String> =
            split_text_by_period(line).into();
        let mut parenthesis_level = 0i32;
        let mut hook_bracket_level = 0i32;
        let mut double_hook_bracket_level = 0i32;
        let mut sentence = String::new();
        while let Some(candidate) = candidates.pop_front() {
            sentence.push_str(&candidate);
            parenthesis_level += candidate.matches('（').count() as i32
                - candidate.matches('）').count() as i32;
            parenthesis_level += candidate.matches('(').count() as i32
                - candidate.matches(')').count() as i32;
            hook_bracket_level += candidate.matches('「').count() as i32
                - candidate.matches('」').count() as i32;
            double_hook_bracket_level += candidate.matches('『').count() as i32
                - candidate.matches('』').count() as i32;
            if parenthesis_level == 0 && hook_bracket_level == 0 && double_hook_bracket_level == 0
            {
                if !sentence.trim().is_empty() {
                    sentences.push(sentence.trim().to_string());
                }
                sentence.clear();
            }
        }
        if !sentence.trim().is_empty() {
            sentences.extend(split_text_by_period(sentence.trim()));
        }
    }
    sentences
}

/// 按 token 数把句子贪心装窗：装不下就开新窗，窗间不重叠
/// （`flat_utils.py:79-120`；stride=-1 的语义就是「切但不重叠」）。
/// 返回（子文档 id，窗文本）。窗文本 = 各句 strip 后直连（无分隔符）。
fn split_flat_document(
    sentences: &[String],
    doc_id: &str,
    doc_text: &str,
    max_token_length: usize,
    tokenizer: &CharTokenizer,
) -> Vec<(String, String)> {
    let sentence_tokens: Vec<usize> =
        sentences.iter().map(|s| tokenizer.tokenize_len(s)).collect();
    if sentence_tokens.iter().sum::<usize>() <= max_token_length {
        return vec![(doc_id.to_string(), doc_text.to_string())];
    }
    let mut sub_documents: Vec<(String, String)> = Vec::new();
    let mut start = 0usize;
    let mut sub_idx = 0usize;
    while start < sentence_tokens.len() {
        let mut end = start;
        let mut total = sentence_tokens[start];
        while end + 1 < sentence_tokens.len()
            && total + sentence_tokens[end + 1] <= max_token_length
        {
            end += 1;
            total += sentence_tokens[end];
        }
        end += 1; // 右开区间
        let window_text: String = sentences[start..end].concat();
        let sub_doc_id = format!("{doc_id}-s{}i{sub_idx}", end - start);
        sub_documents.push((sub_doc_id, window_text));
        start = end;
        sub_idx += 1;
    }
    sub_documents
}

fn orig_doc_pattern() -> &'static Regex {
    static PAT: OnceLock<Regex> = OnceLock::new();
    PAT.get_or_init(|| {
        Regex::new(r"^([a-zA-Z0-9_\-]+?)-s(\d+?)i(\d+?)$").expect("子文档 id 正则编译失败")
    })
}

/// `to_orig_doc_id`（`sub_document.py:14-19`）：剥掉 `-s{stride}i{idx}` 窗后缀。
pub fn to_orig_doc_id(doc_id: &str) -> String {
    match orig_doc_pattern().captures(doc_id) {
        Some(caps) => caps[1].to_string(),
        None => doc_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_raw_text_reads_did_and_strips() {
        // Python: '# D-ID:c-CH-0001'.split()[1].split(':')[1] == 'c-CH-0001'
        let (doc_id, text) = parse_raw_text("# D-ID:c-CH-0001\n　さっきまで、水瀬が使っていたもの。\n");
        assert_eq!(doc_id, "c-CH-0001");
        // strip 只去两端空白（含全角空格），中间不动
        assert_eq!(text, "さっきまで、水瀬が使っていたもの。");
        // 无 D-ID 行 → doc_id 留空，由 add_doc_ids 补
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
        // 以下期望值全部来自 Python 原实现（kwja.utils.flat_senter.split_document）实测
        assert_eq!(split_document("「ああ」。いい。"), vec!["「ああ」。", "いい。"]);
        // 括号跨句末符未闭合 → 候选往后接
        assert_eq!(split_document("「ああ。いい。」"), vec!["「ああ。いい。」"]);
        // 连续句末符算一个（正则的 + 号）
        assert_eq!(split_document("すごい！！本当？"), vec!["すごい！！", "本当？"]);
        // 行尾仍不闭合 → 剩余小段再按句末符切（flat_senter.py:57-58）
        assert_eq!(split_document("「ああ。いい"), vec!["「ああ。", "いい"]);
        assert_eq!(split_document("（あ。い。"), vec!["（あ。", "い。"]);
    }
}
