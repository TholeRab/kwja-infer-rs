//! NFKC 归一化 + TRANSLATION_TABLE，与 Python 侧逐字符一致。
//!
//! 两步与 Python 参考实现同序：先 NFKC 归一化，再过 TRANSLATION_TABLE。

use crate::translation_table::translate;
use unicode_normalization::UnicodeNormalization;

/// 归一化一段文本。
#[must_use]
pub fn normalize_text(text: &str) -> String {
    text.nfkc().filter_map(translate).collect()
}
