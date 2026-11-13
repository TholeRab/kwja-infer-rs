//! word 侧分词：deberta-v2-base-japanese 的 tokenizer.json（vocab 32000）。
//!
//! 与 Python 的对齐点：
//! 1. 主编码：词素 surf 列表按 is_split_into_words=True 编码，add_special_tokens
//!    默认开（调用方没传该参，transformers 默认 True，[CLS] 与 [SEP] 在内）；
//! 2. special 编码：7 个 special token（配置里的 8 个滤掉空格）单独编码，
//!    add_special_tokens=False；
//! 3. merge：主编码 + special + 补 pad 到 max_seq_length（=256）。

use std::collections::HashSet;
use std::path::Path;

use tokenizers::{AddedToken, Tokenizer};

use crate::EncodedInput;

pub struct WordTokenizer {
    tokenizer: Tokenizer,
    pad_id: u32,
    max_seq_length: usize,
    special_ids: Vec<i64>,
}

impl WordTokenizer {
    /// `additional_special_tokens` 是配置里的 8 个（含 `" "`，决定 tokenizer 怎么吃空格）；
    /// `special_tokens` 是滤掉 `" "` 之后的 7 个（追加在主编码之后）。
    pub fn load(
        tokenizer_json: impl AsRef<Path>,
        additional_special_tokens: &[&str],
        special_tokens: &[&str],
        max_seq_length: usize,
    ) -> Result<Self, String> {
        let mut tokenizer =
            Tokenizer::from_file(tokenizer_json.as_ref()).map_err(|e| e.to_string())?;
        let added: Vec<AddedToken> = additional_special_tokens
            .iter()
            .map(|t| AddedToken::from(*t, true))
            .collect();
        tokenizer.add_special_tokens(&added);
        let pad_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or_else(|| "tokenizer.json 里没有 [PAD]".to_string())?;
        let special_encoding = tokenizer
            .encode(special_tokens.to_vec(), false)
            .map_err(|e| e.to_string())?;
        let special_ids: Vec<i64> = special_encoding
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();
        Ok(Self {
            tokenizer,
            pad_id,
            max_seq_length,
            special_ids,
        })
    }

    /// 一句的词素 surf 列表在当前 tokenizer 下的 token 数
    /// （add_special_tokens=False、is_split_into_words=True：Vec 输入按词预切分，
    /// 逐词独立编码，与 Python 的 .tokens() 长度同源）。
    pub fn surf_token_len(&self, surfs: &[String]) -> Result<usize, String> {
        let encoding = self
            .tokenizer
            .encode(surfs.to_vec(), false)
            .map_err(|e| e.to_string())?;
        Ok(encoding.get_ids().len())
    }

    /// 编码一个窗口的词素 surf 列表，输出与 Python `Encoding.merge` 后一致。
    pub fn encode_words(&self, surfs: &[String]) -> Result<EncodedInput, String> {
        let encoding = self
            .tokenizer
            .encode(surfs.to_vec(), true)
            .map_err(|e| e.to_string())?;
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let remaining = self
            .max_seq_length
            .checked_sub(input_ids.len() + self.special_ids.len())
            .ok_or_else(|| {
                format!(
                    "窗口过长：{} 个词素编出 {} 个 id，超出 {} - {}",
                    surfs.len(),
                    input_ids.len(),
                    self.max_seq_length,
                    self.special_ids.len()
                )
            })?;
        input_ids.extend_from_slice(&self.special_ids);
        attention_mask.extend(std::iter::repeat_n(1, self.special_ids.len()));
        input_ids.extend(std::iter::repeat_n(self.pad_id as i64, remaining));
        attention_mask.extend(std::iter::repeat_n(0, remaining));
        Ok(EncodedInput {
            input_ids,
            attention_mask,
        })
    }

    /// 编码一个窗口并保留 word_ids（subword_map 要用）。
    ///
    /// 与 Python 完全同构：主编码（add_special_tokens=True，含 [CLS] 与 [SEP]）
    /// 加 7 个 special，再 pad 到 max_seq_length。special/pad 位的 word_id 是
    /// None；num_tokens 是主编码长度（special 的 token 级下标 = num_tokens + i）。
    pub fn encode_window(&self, surfs: &[String]) -> Result<WindowEncoding, String> {
        let encoding = self
            .tokenizer
            .encode(surfs.to_vec(), true)
            .map_err(|e| e.to_string())?;
        let num_tokens = encoding.get_ids().len();
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let mut word_ids: Vec<Option<usize>> = encoding
            .get_word_ids()
            .iter()
            .map(|w| w.map(|x| x as usize))
            .collect();
        let remaining = self
            .max_seq_length
            .checked_sub(num_tokens + self.special_ids.len())
            .ok_or_else(|| {
                format!(
                    "窗口过长：{} 个词素编出 {} 个 id，超出上限",
                    surfs.len(),
                    num_tokens
                )
            })?;
        input_ids.extend_from_slice(&self.special_ids);
        attention_mask.extend(std::iter::repeat_n(1, self.special_ids.len()));
        word_ids.extend(std::iter::repeat_n(None, self.special_ids.len()));
        input_ids.extend(std::iter::repeat_n(self.pad_id as i64, remaining));
        attention_mask.extend(std::iter::repeat_n(0, remaining));
        word_ids.extend(std::iter::repeat_n(None, remaining));
        Ok(WindowEncoding {
            input_ids,
            attention_mask,
            word_ids,
            num_tokens,
            max_seq_length: self.max_seq_length,
            num_specials: self.special_ids.len(),
        })
    }

    /// 单个 token id 的解码（读音归并要逐 token 表层）。
    pub fn decode_one(&self, id: i64) -> String {
        self.tokenizer
            .decode(&[id.max(0) as u32], false)
            .unwrap_or_default()
    }
}

/// 一个窗口的完整编码（ids + mask + word_ids）。
pub struct WindowEncoding {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    /// 每 token 的词下标（special/pad 为 None）。
    pub word_ids: Vec<Option<usize>>,
    /// 主编码长度（含 [CLS]/[SEP]）。
    pub num_tokens: usize,
    max_seq_length: usize,
    num_specials: usize,
}

impl WindowEncoding {
    /// subword_map[词素][token]（含 special 行列）。
    ///
    /// 第 m 行 = 该词素覆盖的 token 列；special 的词素级下标 = num_morphemes + i、
    /// token 级下标 = num_tokens + i，一一对应打上。
    pub fn subword_map(
        &self,
        num_morphemes: usize,
        include_special_tokens: bool,
    ) -> Vec<Vec<bool>> {
        let s = self.max_seq_length;
        let mut map = vec![vec![false; s]; s];
        let special_token_level: HashSet<usize> =
            ((self.num_tokens)..(self.num_tokens + self.num_specials)).collect();
        for (token_index, word_id) in self.word_ids.iter().enumerate() {
            let Some(w) = word_id else { continue };
            if special_token_level.contains(&token_index) {
                continue;
            }
            if *w < s {
                map[*w][token_index] = true;
            }
        }
        if include_special_tokens {
            for i in 0..self.num_specials {
                let token_index = self.num_tokens + i;
                let morpheme_index = num_morphemes + i;
                if morpheme_index < s && token_index < s {
                    map[morpheme_index][token_index] = true;
                }
            }
        }
        map
    }

    /// FIRST 池化的取数下标：该词素行第一个为真的 token；
    /// 全空行（special 之后的补位行）与 torch 的 `argmax(全零)` 一致取 0。
    pub fn subword_first_token(&self, morpheme: usize, num_morphemes: usize) -> usize {
        if morpheme >= num_morphemes && morpheme < num_morphemes + self.num_specials {
            // special 行：它的 token 就是 num_tokens + i
            return self.num_tokens + (morpheme - num_morphemes);
        }
        for (t, w) in self.word_ids.iter().enumerate() {
            if *w == Some(morpheme) {
                return t;
            }
        }
        0
    }
}
