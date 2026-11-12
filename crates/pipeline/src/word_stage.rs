//! word 阶段入口（阶段 6 落地：解析＋分窗；Python 侧
//! `datamodule/datasets/flat_word_document.py` 的对应物）。
//!
//! 输入是 char 阶段的产出（Juman++ 文本），先解析成扁平文档，
//! 再按 token 数切成 stride=2 的滑窗。★sid 直接取自注释行（不重编）——
//! 编错了整章都挂错对象。分窗必须复用上游 `SequenceSplitter`（见 window.rs
//! 模块注释：窗间大量重叠，不是「装不下才开新窗」的贪心）。
//!
//! 目标句＝每窗末尾 stride 句：被滑窗带进来的前缀句只做上下文，
//! 不写回结果（阶段 7 的 writer 里按目标句过滤）。

use crate::jumanpp;
use crate::window::{SequenceSplitter, target_sentence_flags};
use kwja_tokenizer::word_side::WordTokenizer;

/// 一句：sid ＋ 词素 surf 列表（surf 供按词预切分计 token）＋ lemma 列表
/// （char 阶段的正規化形，阶段 7 里作 norm 查词典/jinf）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSentence {
    pub sid: String,
    pub surfs: Vec<String>,
    pub lemmas: Vec<String>,
}

/// 扁平文档：doc_id ＋ 句子序列。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordDocument {
    pub doc_id: String,
    pub sentences: Vec<WordSentence>,
}

/// 一个分窗：子文档 id ＋ 切进的句子 ＋ 在原文档里的句子起点下标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordWindow {
    pub doc_id: String,
    pub sentences: Vec<WordSentence>,
    pub first_sentence_index: usize,
}

impl WordWindow {
    /// 窗内全部词素 surf（与 Python `document.morphemes` 同序）。
    pub fn surfs(&self) -> Vec<String> {
        self.sentences
            .iter()
            .flat_map(|s| s.surfs.iter().cloned())
            .collect()
    }

    /// 窗内全部词素 lemma（norm）。
    pub fn lemmas(&self) -> Vec<String> {
        self.sentences
            .iter()
            .flat_map(|s| s.lemmas.iter().cloned())
            .collect()
    }

    /// 每句是否目标句（`target_sentence_flags`，按本窗 doc_id 的 stride 判）。
    pub fn target_sentence_flags(&self) -> Vec<bool> {
        target_sentence_flags(&self.doc_id, self.sentences.len())
    }
}

/// 一段 Juman++ 文本 → 扁平文档（`flat_word_document.parse_jumanpp_document`；
/// sid 取自注释行）。
pub fn parse_jumanpp_document(text: &str, doc_id: &str) -> WordDocument {
    let mut sentences: Vec<WordSentence> = Vec::new();
    for sent_text in jumanpp::split_sentences(text) {
        let parsed = jumanpp::parse_sentence(&sent_text);
        sentences.push(WordSentence {
            sid: parsed.sid,
            surfs: parsed.morphemes.iter().map(|m| m.surf.clone()).collect(),
            lemmas: parsed.morphemes.iter().map(|m| m.lemma.clone()).collect(),
        });
    }
    WordDocument {
        doc_id: doc_id.to_string(),
        sentences,
    }
}

/// 按 token 数把文档切成子文档（`flat_word_document.split_flat_word_document`，
/// 对照上游 datasets/base.py:90-108 ＋ word_inference.py 的 max_token_length）。
///
/// 句长按**逐词素**预切分计 token（`_token_len`：add_special_tokens=False、
/// is_split_into_words=True）——词素边界与字符边界不同，整句切会算错。
/// 总长不超上限时整文档就是一窗（doc_id 不加后缀，目标句判定走「全 True」分支）。
pub fn split_flat_word_document(
    document: &WordDocument,
    max_token_length: usize,
    tokenizer: &WordTokenizer,
    stride: i64,
) -> Result<Vec<WordWindow>, String> {
    let mut seq_lens: Vec<usize> = Vec::with_capacity(document.sentences.len());
    for sentence in &document.sentences {
        seq_lens.push(tokenizer.surf_token_len(&sentence.surfs)?);
    }
    if seq_lens.iter().sum::<usize>() <= max_token_length {
        return Ok(vec![WordWindow {
            doc_id: document.doc_id.clone(),
            sentences: document.sentences.clone(),
            first_sentence_index: 0,
        }]);
    }

    let splitter = SequenceSplitter::new(seq_lens, max_token_length, stride);
    let mut subs: Vec<WordWindow> = Vec::new();
    for (sub_idx, span) in splitter.split_into_spans().into_iter().enumerate() {
        let sentences = document.sentences[span.start..span.end].to_vec();
        // to_sub_doc_id（sub_document.py:10-11）：`{doc_id}-s{stride}i{sub_idx}`
        let doc_id = format!("{}-s{}i{sub_idx}", document.doc_id, span.stride);
        subs.push(WordWindow {
            doc_id,
            sentences,
            first_sentence_index: span.start,
        });
    }
    Ok(subs)
}
