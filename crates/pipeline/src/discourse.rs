//! 談話関係解码（`flat_discourse.py` 的移植）。
//!
//! 节 = 带 節-区切 的基本句；行/列坐标都是节末基本句**第一个词素**的
//! 窗内下标（不是 head——head 有优先级选择，两者 7.1% 不同）。
//! Python 侧先对整张 (seq, seq, rel) 做 softmax＋阈值回落，这里按需算
//! 单个 (i, j) 对，概率逐元素相同。

/// 单对 logits → 关系串（softmax → argmax → 低于阈值改判「談話関係なし」）。
fn decode_pair(logits: &[f32], none_index: usize, threshold: f32) -> usize {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let mut best = 0usize;
    let mut best_p = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let p = (v - max).exp() / sum;
        // torch max(dim) 平局取首个；softmax 保序，这里同样取首个
        if p > best_p {
            best_p = p;
            best = i;
        }
    }
    if best_p < threshold {
        none_index
    } else {
        best
    }
}

/// `add_discourse_plain`。
///
/// `pair_logits(i, j)` 返回第 i 节对第 j 节的 7 维 logits（未 softmax）。
pub fn add_discourse(
    clause_m0_gi: &[usize],
    clause_end_index: &[usize],
    clause_sid: &[String],
    pair_logits: &dyn Fn(usize, usize) -> Vec<f32>,
    discourse_relations: &[String],
    threshold: f32,
) -> Vec<(usize, String)> {
    let none_index = discourse_relations
        .iter()
        .position(|r| r == "談話関係なし")
        .unwrap_or(0);
    let m = clause_m0_gi.len();
    let mut out = Vec::new();
    for i in 0..m {
        let mut buf: Vec<String> = Vec::new();
        for j in 0..m {
            let logits = pair_logits(clause_m0_gi[i], clause_m0_gi[j]);
            let rel = &discourse_relations[decode_pair(&logits, none_index, threshold)];
            if rel != "談話関係なし" {
                buf.push(format!("{}/{}/{}", clause_sid[j], clause_end_index[j], rel));
            }
        }
        if !buf.is_empty() {
            out.push((i, buf.join(";")));
        }
    }
    out
}
