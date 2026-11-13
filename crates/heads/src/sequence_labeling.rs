//! 逐 token 分类头的解码：pos／subpos／conjtype／conjform 的掩码 argmax
//! 与 NE 头的 CRF Viterbi，均逐 id 复刻 Python 参考实现。
//!
//! 三处必须保持的解码细节：
//! 1. subpos／conjform 的 argmax 不能直接取：先用 pos（conjtype）结果把非法标签的
//!    logits 掩成 `MASKED`（-1024.0）再取；
//! 2. 非活用词素：conjtype 改判 `"*"`，conjform 直接取表值 `CONJFORM_STAR_ID`
//!    ——漏掉这个改判，下游解析的活用形就会错；
//! 3. CRF 的回溯区间是 `history[head-min : tail-min][::-1]`，掩码外用 `"O"` 填充。

use crate::mask_tables::{
    CONJTYPE_CONJFORM_ALLOWED, CONJFORM_TAGS, CONJFORM_STAR_ID, CONJTYPE_STAR_INDEX, INFLECTABLE,
    MASKED, POS_SUBPOS_ALLOWED, SUBPOS_TAGS,
};

/// numpy／torch 语义的 argmax：并列时取第一个最大值的下标。
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max {
            max = v;
            best = i;
        }
    }
    best
}

/// 四路词素属性预测（下标语义，与 Python 的 `list[int]` 一一对应）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphemeAttributePredictions {
    pub pos: Vec<usize>,
    pub subpos: Vec<usize>,
    pub conjtype: Vec<usize>,
    pub conjform: Vec<usize>,
}

/// 四路词素属性的掩码 argmax 预测。
///
/// # Panics
/// 四路 logits 行数不一致、或掩码后 argmax 落在非法标签上时 panic——
/// 那属于程序错误而非数据错误。
pub fn get_morpheme_attribute_predictions(
    pos_logits: &[Vec<f32>],
    subpos_logits: &[Vec<f32>],
    conjtype_logits: &[Vec<f32>],
    conjform_logits: &[Vec<f32>],
) -> MorphemeAttributePredictions {
    assert_eq!(pos_logits.len(), subpos_logits.len(), "zip(strict=True)");
    assert_eq!(pos_logits.len(), conjtype_logits.len(), "zip(strict=True)");
    assert_eq!(pos_logits.len(), conjform_logits.len(), "zip(strict=True)");

    let pos: Vec<usize> = pos_logits.iter().map(|row| argmax(row)).collect();
    let mut conjtype: Vec<usize> = conjtype_logits.iter().map(|row| argmax(row)).collect();
    let mut subpos = Vec::with_capacity(pos.len());
    let mut conjform = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        let pos_index = pos[i];
        let allowed_subpos = POS_SUBPOS_ALLOWED[pos_index];
        let mut masked_row: Vec<f32> = subpos_logits[i].clone();
        for (subpos_index, _) in SUBPOS_TAGS.iter().enumerate() {
            if !allowed_subpos.contains(&subpos_index) {
                masked_row[subpos_index] = MASKED;
            }
        }
        let subpos_index = argmax(&masked_row);
        assert!(
            allowed_subpos.contains(&subpos_index),
            "subpos argmax 落在非法标签上 (token {i})"
        );
        subpos.push(subpos_index);

        let conjtype_index = conjtype[i];
        if INFLECTABLE.contains(&(pos_index, subpos_index)) {
            let allowed_conjform = CONJTYPE_CONJFORM_ALLOWED[conjtype_index];
            let mut masked_row: Vec<f32> = conjform_logits[i].clone();
            for (conjform_index, _) in CONJFORM_TAGS.iter().enumerate() {
                if !allowed_conjform.contains(&conjform_index) {
                    masked_row[conjform_index] = MASKED;
                }
            }
            let conjform_index = argmax(&masked_row);
            assert!(
                allowed_conjform.contains(&conjform_index),
                "conjform argmax 落在非法标签上 (token {i})"
            );
            conjform.push(conjform_index);
        } else {
            conjtype[i] = CONJTYPE_STAR_INDEX;
            conjform.push(CONJFORM_STAR_ID);
        }
    }
    MorphemeAttributePredictions { pos, subpos, conjtype, conjform }
}

/// CRF 的模型参数（推理时从权重里读；这里作为纯函数的输入）。
#[derive(Debug, Clone, PartialEq)]
pub struct CrfParameters {
    pub start_transitions: Vec<f32>,
    /// `transitions[prev][next]`
    pub transitions: Vec<Vec<f32>>,
    pub end_transitions: Vec<f32>,
    /// NE_TAGS 里 `"O"` 的下标（掩码外的填充标签）。
    pub o_index: usize,
}

/// CRF Viterbi 解码。
///
/// `emissions[b][j][t]`、`mask[b][j]`。掩码外的 token 填 `"O"`。
/// `max(dim=1)` 的并列取首与 `score[i].max(dim=0)` 的并列取首都由 [`argmax`] 保证。
pub fn viterbi_decode(
    emissions: &[Vec<Vec<f32>>],
    mask: &[Vec<bool>],
    params: &CrfParameters,
) -> Vec<Vec<usize>> {
    let batch_size = emissions.len();
    assert_eq!(batch_size, mask.len(), "emissions 与 mask 的 batch 不一致");
    let seq_len = mask[0].len();
    let num_tags = params.start_transitions.len();

    // head = where(mask, arange, seq_len-1).amin()：第一个真位；全假时 seq_len-1
    let head_indices: Vec<usize> = mask
        .iter()
        .map(|m| m.iter().position(|&x| x).unwrap_or(seq_len - 1))
        .collect();
    // tail = (arange * mask).amax()：最后一个真位；全假时 0
    let tail_indices: Vec<usize> = mask
        .iter()
        .map(|m| m.iter().rposition(|&x| x).unwrap_or(0))
        .collect();
    let min_head_index = *head_indices.iter().min().unwrap();
    let max_tail_index = *tail_indices.iter().max().unwrap();

    let mut score: Vec<Vec<f32>> = (0..batch_size)
        .map(|b| {
            let h = head_indices[b];
            (0..num_tags)
                .map(|t| params.start_transitions[t] + emissions[b][h][t])
                .collect()
        })
        .collect();
    let mut history: Vec<Vec<usize>> = Vec::new();
    for j in (min_head_index + 1)..=max_tail_index {
        let mut next_scores = vec![vec![0.0f32; num_tags]; batch_size];
        let mut argmax_prev = vec![0usize; batch_size * num_tags];
        for b in 0..batch_size {
            for t in 0..num_tags {
                let mut best_value = f32::NEG_INFINITY;
                let mut best_prev = 0usize;
                for (prev, (prev_score, trans_row)) in
                    score[b].iter().zip(params.transitions.iter()).enumerate()
                {
                    let v = prev_score + trans_row[t] + emissions[b][j][t];
                    if v > best_value {
                        best_value = v;
                        best_prev = prev;
                    }
                }
                next_scores[b][t] = best_value;
                argmax_prev[b * num_tags + t] = best_prev;
            }
        }
        for b in 0..batch_size {
            if mask[b][j] && head_indices[b] != j {
                score[b].copy_from_slice(&next_scores[b]);
            }
        }
        history.push(argmax_prev);
    }
    for s in &mut score {
        for (t, end) in s.iter_mut().zip(params.end_transitions.iter()) {
            *t += end;
        }
    }

    let mut out = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let head_index = head_indices[b];
        let tail_index = tail_indices[b];
        let mut best_tag = argmax(&score[b]);
        let mut best_tags = vec![best_tag];
        // history[head-min : tail-min][::-1]：从 tail 回溯到 head
        let lo = head_index - min_head_index;
        let hi = tail_index - min_head_index;
        for k in (lo..hi).rev() {
            let step = &history[k];
            best_tag = step[b * num_tags + best_tag];
            best_tags.push(best_tag);
        }
        for _ in 0..head_index {
            best_tags.push(params.o_index);
        }
        best_tags.reverse();
        for _ in 0..seq_len.saturating_sub(tail_index + 1) {
            best_tags.push(params.o_index);
        }
        assert_eq!(best_tags.len(), seq_len, "解码长度与 seq_len 不一致");
        out.push(best_tags);
    }
    out
}
