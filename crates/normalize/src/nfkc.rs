//! NFKC 归一化 + TRANSLATION_TABLE，与 Python 侧逐字符一致。
//!
//! Python 依据：`cli.py:293-299` 与 `utils/normalization.py:7-8` 同逻辑——
//! 先 `unicodedata.normalize("NFKC", text)`，再 `.translate(TRANSLATION_TABLE)`。

use crate::translation_table::translate;
use unicode_normalization::UnicodeNormalization;

/// 归一化一段文本。
#[must_use]
pub fn normalize_text(text: &str) -> String {
    text.nfkc().filter_map(translate).collect()
}
