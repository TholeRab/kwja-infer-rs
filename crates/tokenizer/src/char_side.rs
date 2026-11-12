//! char 侧分词：逐 id 复刻 transformers 5.18 `BertJapaneseTokenizer`（慢实现）。
//!
//! char-wwm 快照里没有 tokenizer.json，只有 vocab.txt；`AutoTokenizer.from_pretrained`
//! 实际返回慢分词器（tokenizer_config.json：tokenizer_class=BertJapaneseTokenizer、
//! do_word_tokenize=True、word_tokenizer_type=basic、subword_tokenizer_type=character、
//! do_lower_case=False、mask_token=" "）。
//! 管线（对照源码位置）：
//! 1. added token 切分：`tokenization_python.py:625-683`（tokens_trie.split → 非 added 段进 _tokenize）；
//! 2. BasicTokenizer：`tokenization_bert_japanese.py:660-696`
//!    （_clean_text 去控制字符/空白归一 → NFC → 空白切分 → 标点逐字隔离；汉 字加空格已关）；
//! 3. CharacterTokenizer：`tokenization_bert_japanese.py:597-622`（词内再 NFKC，逐字符查 vocab，缺字 [UNK]）；
//! 4. 组装：[CLS] + ids + [SEP]，右侧补 [PAD] 到 max_seq_length。

use std::collections::{HashMap, HashSet};
use std::path::Path;

use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

use crate::EncodedInput;

/// all_special_tokens（BertJapaneseTokenizer：unk/sep/pad/cls/mask；mask_token 恰为 " "）。
const SPECIAL_TOKENS: [&str; 5] = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", " "];

pub struct CharTokenizer {
    vocab: HashMap<String, u32>,
    /// 新增（不在 base vocab）的 added token → id，从 len(vocab) 起编号。
    added_ids: HashMap<String, u32>,
    /// trie 切分用：specials + 额外 added，按长度降序（最左最长匹配）。
    trie_tokens: Vec<String>,
    never_split: HashSet<String>,
    pad_id: u32,
    cls_id: u32,
    sep_id: u32,
    unk_id: u32,
    max_seq_length: usize,
}

impl CharTokenizer {
    /// `vocab_txt`：快照里的 vocab.txt；`additional_special_tokens`：配置里的额外 special
    /// （char 侧为 [" "]，经 transformers `add_special_tokens` 变成 AddedToken(normalized=False, special=True)）。
    pub fn load(
        vocab_txt: impl AsRef<Path>,
        additional_special_tokens: &[&str],
        max_seq_length: usize,
    ) -> Result<Self, String> {
        let content = std::fs::read_to_string(vocab_txt.as_ref())
            .map_err(|e| format!("读 vocab.txt 失败：{e}"))?;
        // Python load_vocab：token.rstrip("\n") 后按行号编 id。
        let vocab: HashMap<String, u32> = content
            .split_inclusive('\n')
            .enumerate()
            .map(|(i, line)| (line.strip_suffix('\n').unwrap_or(line).to_string(), i as u32))
            .collect();
        let get = |t: &str| vocab.get(t).copied();
        let pad_id = get("[PAD]").ok_or("vocab.txt 里没有 [PAD]")?;
        let cls_id = get("[CLS]").ok_or("vocab.txt 里没有 [CLS]")?;
        let sep_id = get("[SEP]").ok_or("vocab.txt 里没有 [SEP]")?;
        let unk_id = get("[UNK]").ok_or("vocab.txt 里没有 [UNK]")?;

        let mut added_ids = HashMap::new();
        let mut next = vocab.len() as u32;
        for t in additional_special_tokens {
            if let Some(id) = get(t) {
                added_ids.insert((*t).to_string(), id);
            } else {
                added_ids.insert((*t).to_string(), next);
                next += 1;
            }
        }

        let mut all: Vec<String> = SPECIAL_TOKENS
            .iter()
            .map(|s| (*s).to_string())
            .chain(additional_special_tokens.iter().map(|s| (*s).to_string()))
            .collect();
        all.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
        all.dedup();
        let never_split: HashSet<String> = all.iter().cloned().collect();

        Ok(Self {
            vocab,
            added_ids,
            trie_tokens: all,
            never_split,
            pad_id,
            cls_id,
            sep_id,
            unk_id,
            max_seq_length,
        })
    }

    /// 编码一段（已归一化的）文本，产出与 Python `tokenizer(text, padding="max_length",
    /// truncation=False, max_length=512)` 完全相同的 ids 与 mask。
    pub fn encode(&self, text: &str) -> Result<EncodedInput, String> {
        let mut tokens: Vec<String> = Vec::new();
        for segment in self.split_on_added(text) {
            if segment.is_empty() {
                continue;
            }
            if self.added_ids.contains_key(&segment) || self.never_split.contains(&segment) {
                tokens.push(segment);
            } else {
                for word in self.basic_tokenize(&segment) {
                    tokens.extend(self.character_tokenize(&word));
                }
            }
        }

        let mut input_ids: Vec<i64> = Vec::with_capacity(tokens.len() + 2);
        input_ids.push(self.cls_id as i64);
        for tok in &tokens {
            let id = self
                .added_ids
                .get(tok)
                .or_else(|| self.vocab.get(tok))
                .unwrap_or(&self.unk_id);
            input_ids.push(*id as i64);
        }
        input_ids.push(self.sep_id as i64);

        let mut attention_mask = vec![1i64; input_ids.len()];
        if input_ids.len() < self.max_seq_length {
            let pad = self.max_seq_length - input_ids.len();
            input_ids.extend(std::iter::repeat_n(self.pad_id as i64, pad));
            attention_mask.extend(std::iter::repeat_n(0, pad));
        }
        Ok(EncodedInput {
            input_ids,
            attention_mask,
        })
    }

    /// tokens_trie.split：从左到右扫，命中最长的 added token 就切出来（保序）。
    fn split_on_added(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut segments: Vec<String> = Vec::new();
        let mut buf = String::new();
        let mut i = 0;
        while i < chars.len() {
            let rest: String = chars[i..].iter().collect();
            match self
                .trie_tokens
                .iter()
                .find(|t| rest.starts_with(t.as_str()))
            {
                Some(t) => {
                    if !buf.is_empty() {
                        segments.push(std::mem::take(&mut buf));
                    }
                    segments.push(t.clone());
                    i += t.chars().count();
                }
                None => {
                    buf.push(chars[i]);
                    i += 1;
                }
            }
        }
        if !buf.is_empty() {
            segments.push(buf);
        }
        segments
    }

    /// BasicTokenizer.tokenize（do_lower_case=False、tokenize_chinese_chars=False、strip_accents=None）。
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = clean_text(text);
        let nfc: String = cleaned.nfc().collect();
        let mut split_tokens: Vec<String> = Vec::new();
        for token in nfc.split_whitespace() {
            split_tokens.extend(self.split_on_punc(token));
        }
        let joined = split_tokens.join(" ");
        joined.split_whitespace().map(String::from).collect()
    }

    /// _run_split_on_punc：never_split 的词整存，其余按标点逐字隔离。
    fn split_on_punc(&self, token: &str) -> Vec<String> {
        if self.never_split.contains(token) {
            return vec![token.to_string()];
        }
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in token.chars() {
            if is_punctuation(c) {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// CharacterTokenizer.tokenize：词内 NFKC 后逐字符查 vocab，缺字 [UNK]。
    fn character_tokenize(&self, word: &str) -> Vec<String> {
        let nfkc: String = word.nfkc().collect();
        nfkc.chars()
            .map(|c| {
                let s = c.to_string();
                if self.vocab.contains_key(&s) {
                    s
                } else {
                    "[UNK]".to_string()
                }
            })
            .collect()
    }
}

/// _clean_text：cp==0、U+FFFD、控制字符删除；空白归一成空格。
fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let u = c as u32;
        if u == 0 || u == 0xFFFD || is_control(c) {
            continue;
        }
        if is_whitespace(c) {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// _is_whitespace：空格/制表/换行/回车 或 Unicode Zs。
fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
        || get_general_category(c) == GeneralCategory::SpaceSeparator
}

/// _is_control：\t\n\r 算空白不算控制；其余 C*（Cc/Cf/Cn/Co/Cs）算控制。
fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    matches!(
        get_general_category(c),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Unassigned
            | GeneralCategory::PrivateUse
            | GeneralCategory::Surrogate
    )
}

/// _is_punctuation：非字母数字的 ASCII 视作标点，其余按 Unicode P*。
fn is_punctuation(c: char) -> bool {
    let u = c as u32;
    if (33..=47).contains(&u)
        || (58..=64).contains(&u)
        || (91..=96).contains(&u)
        || (123..=126).contains(&u)
    {
        return true;
    }
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}
