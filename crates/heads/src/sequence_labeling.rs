//! 逐 token 分类头（阶段 3 落地）：pos / subpos / conjtype / conjform / ne 等。
//! subpos 与 conjform 不能直接 argmax：先用 pos（conjtype）结果把非法标签掩成 -1024.0 再取。
