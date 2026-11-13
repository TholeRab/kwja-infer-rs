//! relation-wise 头的解码：dependency 的 topk 与
//! discourse 的 softmax 加阈值回落。
//!
//! discourse 的阈值取 0.0；
//! 配置值为 0.0 时阈值分支实际不会触发（softmax 概率非负），但回落逻辑必须保留——
//! 配置一变它就是行为的一部分。

/// torch.topk 的下标部分：按值降序取前 k 个下标。
///
/// Rust 稳定排序在并列时保留下标序；随机 logits 无并列，与 torch 行为一致。
#[must_use]
pub fn topk_indices(row: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}

/// (batch, seq, seq) 的 dependency logits：每个 token 取前 k 个母节候选。
#[must_use]
pub fn topk_dim2(logits: &[Vec<Vec<f32>>], k: usize) -> Vec<Vec<Vec<usize>>> {
    logits
        .iter()
        .map(|matrix| matrix.iter().map(|row| topk_indices(row, k)).collect())
        .collect()
}

/// discourse 头：softmax(dim=3) 后取最大概率，低于阈值改判「談話関係なし」。
///
/// `logits[b][i][j][rel]`；`none_index` 是 DISCOURSE_RELATIONS 里「談話関係なし」的下标。
#[must_use]
pub fn discourse_decode(
    logits: &[Vec<Vec<Vec<f32>>>],
    threshold: f32,
    none_index: usize,
) -> Vec<Vec<Vec<usize>>> {
    logits
        .iter()
        .map(|matrix| {
            matrix
                .iter()
                .map(|rows| {
                    rows.iter()
                        .map(|row| decode_row(row, threshold, none_index))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn decode_row(row: &[f32], threshold: f32, none_index: usize) -> usize {
    let max_value = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&v| (v - max_value).exp()).sum();
    let mut best_index = 0usize;
    let mut best_prob = -1.0f32;
    for (i, &v) in row.iter().enumerate() {
        let p = (v - max_value).exp() / sum;
        if p > best_prob {
            best_prob = p;
            best_index = i;
        }
    }
    if best_prob < threshold {
        none_index
    } else {
        best_index
    }
}
