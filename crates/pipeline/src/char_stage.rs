//! char 阶段：RAW 章节文本到 Juman++ 文本，与 Python 参考链逐字节对齐。
//!
//! 数据流（对照 Python 侧）：
//! 1. `# D-ID:` 解析与原文拼接；
//! 2. doc_id 补齐：{prefix}-{按文档数零填充的序号}；
//! 3. 正则分句；
//! 4. 按 token 数贪心切窗，窗间不重叠
//!    （max_token_length = max_seq_length - len(extra_special_tokens) - 2）；
//! 5. 每窗：tokenize（[CLS]+ids+[SEP]+pad）、char encoder、三头逐位 argmax、
//!    非 special 位过滤、标签、分句/分词/归一化，得到 Juman++ 行。
//!
//! sid 口径：`# S-ID:{orig_doc_id}-{sid} kwja:{version}`，sid 从 1 起、
//! 跨窗连续、orig_doc_id 变化时重置。

use std::path::Path;
use std::sync::OnceLock;

use kwja_encoder::{CharEncoderInput, Encoder};
use kwja_tokenizer::char_side::CharTokenizer;
use regex::Regex;

use crate::char_tags::{
    convert_char_predictions_into_tags, sentence_jumanpp, set_morphemes_plain, set_sentences_plain,
};

/// 写入 S-ID 行的 kwja 版本号。
pub const KWJA_VERSION: &str = "2.6.0";

/// char 阶段 tokenizer 的附加 special token：一个半角空格。
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
        // （extra 即 1 个空格 special）
        let extra_special_tokens = CHAR_ADDITIONAL_SPECIAL_TOKENS.len();
        Ok(Self {
            tokenizer,
            max_seq_length,
            extra_special_tokens,
        })
    }

    /// 跑完整个 char 阶段，返回 Juman++ 文本。
    ///
    /// texts：每章一段已归一化（NFKC 加转义表）的原文——CLI 流程里
    /// normalize_text 发生在写输入文件时，归一化不属于本模块。
    /// doc_id_prefix：doc_id 为空时的前缀；Python 默认取时间戳，Rust 侧
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

        // 3-4. 正则分句 + 按 token 数切窗
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

        // 5. 逐窗前向 + 拼 Juman++
        let special_ids = self.tokenizer.writer_special_ids();
        let mut output = String::new();
        let mut prev_doc_id = String::new();
        let mut prev_sid = 0u64;
        for (doc_id, text) in &windows {
            let encoding = self.tokenizer.encode(text)?;
            // 超长窗直接跳过（参考实现先打 warning 再 continue，这里只 continue）
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

/// 解析一段原文：`# D-ID:` 注释行取 doc_id，其余行拼回正文并 strip。
/// strip 必须照抄参考实现：模型标签按 strip 后的长度给；取 id 的写法
/// 也照抄其 split 语义。格式残缺行按普通原文行处理（语料里不出现）。
pub fn parse_raw_text(text: &str) -> (String, String) {
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

/// 按句末符把一行切成候选段。
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

/// 正则分句：先按行切、再按句末符切、再用三种括号配平把候选合回去。
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
/// （stride=-1 的语义就是「切但不重叠」）。
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

/// 剥掉 -s{stride}i{idx} 窗后缀，还原原始 doc_id。
pub fn to_orig_doc_id(doc_id: &str) -> String {
    match orig_doc_pattern().captures(doc_id) {
        Some(caps) => caps[1].to_string(),
        None => doc_id.to_string(),
    }
}
