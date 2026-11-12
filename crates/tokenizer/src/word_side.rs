//! word 侧分词：deberta-v2-base-japanese 的 tokenizer.json（vocab 32000）。
//!
//! 与 Python 的对齐点（`datamodule/datasets/word_inference.py:79-143`）：
//! 1. 主编码：词素 surf 列表按 `is_split_into_words=True` 编码，add_special_tokens
//!    **默认开**（上游没传该参，transformers 默认 True → [CLS]…[SEP] 在内）；
//! 2. special 编码：7 个 special token（配置里的 8 个滤掉 `" "`）单独编码，
//!    `add_special_tokens=False`；
//! 3. merge：主编码 + special + 补 pad 到 max_seq_length（=256）。

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
        let special_ids: Vec<i64> = special_encoding.get_ids().iter().map(|&i| i as i64).collect();
        Ok(Self {
            tokenizer,
            pad_id,
            max_seq_length,
            special_ids,
        })
    }

    /// 一句的词素 surf 列表在当前 tokenizer 下的 token 数
    /// （`flat_word_document._token_len`：add_special_tokens=False、
    /// is_split_into_words=True——Vec<String> 输入即按词预切分，
    /// 逐词独立编码，与 Python 的 `.tokens()` 长度同源）。
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
}
