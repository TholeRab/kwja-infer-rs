//! 三张掩码表（阶段 3 落地）：由 Python 侧 `callbacks/utils.py` 一次性导出成静态数组，不在 Rust 里重新推导。
//! 规模实测：POS_TAG_SUBPOS_TAG2SUBPOS_ID 42 条；CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID 639 条；INFLECTABLE 7 条。
