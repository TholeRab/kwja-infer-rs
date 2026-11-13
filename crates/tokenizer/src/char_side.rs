//! char 侧分词：逐 id 复刻 transformers 5.18 BertJapaneseTokenizer（慢实现）。
//!
//! char-wwm 快照里没有 tokenizer.json，只有 vocab.txt；AutoTokenizer.from_pretrained
//! 实际返回慢分词器（tokenizer_config.json：tokenizer_class=BertJapaneseTokenizer、
//! word_tokenizer_type=basic、subword_tokenizer_type=character、do_lower_case=False）。
//! 推理配置传 do_word_tokenize: false，BasicTokenizer 不参与，控制字符/空白/标点预处理都不做。
//! 管线：
//! 1. added token 切分：按 trie 最左最长匹配，非 added 段进 _tokenize；
//! 2. do_word_tokenize=false 时整段直通，不做词级切分；
//! 3. 整段 NFKC 归一化后逐字符查 vocab，缺字记 [UNK]；
//! 4. 组装：[CLS] + ids + [SEP]，右侧补 [PAD] 到 max_seq_length。

use std::collections::{HashMap, HashSet};
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::EncodedInput;

/// all_special_tokens（BertJapaneseTokenizer：unk/sep/pad/cls/mask，即 [MASK]；
/// 另有配置追加的 " "，经 load 的 additional_special_tokens 进 trie）。
/// 取自快照 special_tokens_map.json 与 vocab.txt 第 5 行，两处都是 `[MASK]`。
const SPECIAL_TOKENS: [&str; 6] = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", " "];

pub struct CharTokenizer {
    vocab: HashMap<String, u32>,
    /// 新增（不在 base vocab）的 added token 到 id 的映射，从 len(vocab) 起编号。
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
            .map(|(i, line)| {
                (
                    line.strip_suffix('\n').unwrap_or(line).to_string(),
                    i as u32,
                )
            })
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
                tokens.extend(self.character_tokenize(&segment));
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

    /// `len(tokenizer.tokenize(text))`：不加 special、不补 pad 的 token 数。
    /// 切窗（`split_flat_document`）按这个数贪心装句。
    pub fn tokenize_len(&self, text: &str) -> usize {
        let mut count = 0usize;
        for segment in self.split_on_added(text) {
            if segment.is_empty() {
                continue;
            }
            if self.added_ids.contains_key(&segment) || self.never_split.contains(&segment) {
                count += 1;
            } else {
                count += self.character_tokenize(&segment).len();
            }
        }
        count
    }

    /// writer 的 special 过滤集：
    /// bos/eos 对本 tokenizer 不存在（None 不入集），实际生效的是 pad/cls/sep/mask。
    /// mask token 取快照 special_tokens_map.json 的 [MASK]（vocab id 4）。
    /// 注意 [UNK] 不在过滤集里——UNK 位保留，照样出标签。
    pub fn writer_special_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.pad_id, self.cls_id, self.sep_id];
        let mask_id = self
            .vocab
            .get("[MASK]")
            .copied()
            .expect("vocab.txt 里没有 [MASK]");
        ids.push(mask_id);
        ids
    }

    /// trie 切分：从左到右扫，命中最长的 added token 就切出来（保序）。
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

    /// 整段 NFKC 后逐字符查 vocab，缺字 [UNK]；
    /// do_word_tokenize=false 时输入是整个非 added 段。
    fn character_tokenize(&self, text: &str) -> Vec<String> {
        let nfkc: String = text.nfkc().collect();
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
