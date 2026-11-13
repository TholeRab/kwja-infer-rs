//! word 阶段入口：解析 char 阶段产出的 Juman++ 文本并按 stride=2 分窗。
//!
//! 输入是 char 阶段的产出（Juman++ 文本），先解析成扁平文档，
//! 再按 token 数切成 stride=2 的滑窗。sid 直接取自注释行（不重编）——
//! 编错了整章都挂错对象。分窗必须复用 SequenceSplitter（见 window.rs
//! 模块注释：窗间大量重叠，不是「装不下才开新窗」的贪心）。
//!
//! 目标句=每窗末尾 stride 句：被滑窗带进来的前缀句只做上下文，
//! 不写回结果（输出组装时按目标句过滤）。

use crate::jumanpp;
use crate::window::{SequenceSplitter, target_sentence_flags};
use kwja_tokenizer::word_side::WordTokenizer;

/// 一句：sid、词素 surf 列表（surf 供按词预切分计 token）与 lemma 列表
/// （char 阶段的正規化形，用于查词典/jinf）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSentence {
    pub sid: String,
    pub surfs: Vec<String>,
    pub lemmas: Vec<String>,
}

/// 扁平文档：doc_id 与句子序列。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordDocument {
    pub doc_id: String,
    pub sentences: Vec<WordSentence>,
}

/// 一个分窗：子文档 id、切进的句子与在原文档里的句子起点下标。
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

/// 把一段 Juman++ 文本解析成扁平文档；sid 取自注释行。
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

/// 从 sid 取 doc_id：去掉末尾的 -句号（c-00-1 得 c-00）。等价于正则取 did 组：
/// 先剥掉末尾的 ASCII 数字串，再剥掉一个连字符。
fn sid_did(sid: &str) -> &str {
    let t = sid.trim_end_matches(|c: char| c.is_ascii_digit());
    t.strip_suffix('-').unwrap_or(t)
}

/// Juman++ 文本按 S-ID 文档前缀切分成多个扁平文档。
///
/// 连续同 doc_id 的句归成一个文档，doc_id 变化（或缺失）即开新文档。
/// word 阶段必须按章切文档后再逐文档滑窗；若把整份输入并成一个文档滑窗，
/// 每个目标句的预测上下文都会改变。
pub fn parse_jumanpp_documents_by_doc(text: &str) -> Vec<WordDocument> {
    let mut documents: Vec<WordDocument> = Vec::new();
    let mut cur: Option<WordDocument> = None;
    let mut prev_did: Option<String> = None;
    for sent_text in jumanpp::split_sentences(text) {
        let parsed = jumanpp::parse_sentence(&sent_text);
        let did = if parsed.sid.is_empty() {
            None
        } else {
            Some(sid_did(&parsed.sid).to_string())
        };
        let sentence = WordSentence {
            sid: parsed.sid,
            surfs: parsed.morphemes.iter().map(|m| m.surf.clone()).collect(),
            lemmas: parsed.morphemes.iter().map(|m| m.lemma.clone()).collect(),
        };
        // 与 Python 逐句同构：cur 在手且（did 变了或 did 缺失）就收上一份
        if cur.is_some() && (prev_did != did || did.is_none()) {
            documents.push(cur.take().expect("cur 已判存在"));
        }
        if cur.is_none() {
            cur = Some(WordDocument {
                doc_id: did.clone().unwrap_or_default(),
                sentences: Vec::new(),
            });
            prev_did = did;
        }
        cur.as_mut()
            .expect("上一分支已保证 cur 在手")
            .sentences
            .push(sentence);
    }
    if let Some(doc) = cur.take() {
        documents.push(doc);
    }
    documents
}
/// 按 token 数把文档切成子文档。
///
/// 句长按逐词素预切分计 token（add_special_tokens=False、is_split_into_words=True）
/// ——词素边界与字符边界不同，整句切会算错。
/// 总长不超上限时整文档就是一窗（doc_id 不加后缀，目标句判定走全 True 分支）。
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
        // 子文档 id：{doc_id}-s{stride}i{sub_idx}
        let doc_id = format!("{}-s{}i{sub_idx}", document.doc_id, span.stride);
        subs.push(WordWindow {
            doc_id,
            sentences,
            first_sentence_index: span.start,
        });
    }
    Ok(subs)
}
