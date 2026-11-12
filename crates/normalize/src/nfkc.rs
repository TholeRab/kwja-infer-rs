//! NFKC 归一化（阶段 2 落地）：用 unicode-normalization 对齐 Python `unicodedata.normalize("NFKC", s)`，
//! 再过 TRANSLATION_TABLE；闸是与 Python 逐 id / 逐字符完全相同。
