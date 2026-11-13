//! word 阶段解码段：13 头解码一个窗，及其纯函数工具（读音归并、依存掩码与 topk、argmax）。

use std::collections::HashMap;

use kwja_heads::forward::{RelationWiseProjection, SelectionProjection};

use crate::chunking::Chunk;
use crate::morpheme_build::FlatMorpheme;

/// 读音归并：token 级读音按 subword_map 归并到词素。
pub(crate) fn get_word_level_readings(
    readings: &[&str],
    tokens: &[String],
    subword_map: &[Vec<bool>],
    unk_id: (&str, &str),
) -> Vec<String> {
    let (unk, id) = unk_id;
    let mut ret = Vec::with_capacity(subword_map.len());
    for flags in subword_map {
        let mut item = String::new();
        for ((token, reading), &flag) in tokens.iter().zip(readings).zip(flags) {
            if flag {
                if *reading == unk || *reading == id {
                    item.push_str(token);
                } else {
                    item.push_str(reading);
                }
            }
        }
        if !item.is_empty() {
            ret.push(item);
        } else if flags.iter().any(|&f| f) {
            ret.push("_".to_string());
        }
    }
    ret
}

/// 依存掩码：目标句的块对角（除对角线）+ [ROOT] 列（扁平化的 seq 行 seq 列）。
pub(crate) fn dependency_mask(ranges: &[(usize, usize)], si: usize, root_gi: usize, seq: usize) -> Vec<bool> {
    let mut mask = vec![false; seq * seq];
    let (lo, hi) = ranges[si];
    for i in lo..hi {
        for j in lo..hi {
            mask[i * seq + j] = i != j;
        }
        mask[i * seq + root_gi] = true;
    }
    mask
}

/// 依存一行的 topk 候选：只对掩码放行的列算 logits（row_gathered 与整行逐元素相同），
/// 掩掉的列保持 MASKED。掩码只放开同句词素加 [ROOT]，
/// 比算满整窗少一个数量级的 gelu，而 topk 结果与算完整行再掩一致。
pub(crate) fn dependency_topk_masked(
    dep_proj: &kwja_heads::forward::SelectionProjection,
    hg: usize,
    mask: &[bool],
    max_seq_length: usize,
    topk: usize,
) -> Vec<usize> {
    let row_mask = &mask[hg * max_seq_length..(hg + 1) * max_seq_length];
    let cols: Vec<usize> = row_mask
        .iter()
        .enumerate()
        .filter(|&(_, ok)| *ok)
        .map(|(j, _)| j)
        .collect();
    let vals = dep_proj.row_gathered(hg, &cols);
    let mut masked = vec![kwja_heads::mask_tables::MASKED; max_seq_length];
    for (j, v) in cols.iter().zip(vals) {
        masked[*j] = v[0];
    }
    topk_indices(&masked, topk)
}

/// torch.topk 的等价物：降序取 k 个下标；并列时取先出现者。
fn topk_indices(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    idx.truncate(k);
    idx
}

/// 把 sid 里的 -s\d+i\d+- 窗标记替换回 -；不能剥 2026xxxxxxxx-N- 这类日期前缀。
pub(crate) fn strip_subdoc(sid: &str) -> String {
    static PAT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pat = PAT.get_or_init(|| {
        regex::Regex::new(r"-s\d+i\d+-").expect("子文档标记正则编译失败")
    });
    pat.replace_all(sid, "-").into_owned()
}

pub(crate) fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// 解码产物（跨窗保留的小数据）：之后的窗的合并体视图要读 owner 窗的这三样。
/// 全量驻留也只有约 0.2 MB/窗；大投影不在此列（随段释放，见 predict_text）。
pub(crate) struct WinKeep {
    pub(crate) chunks: Vec<Chunk>,
    pub(crate) sids: Vec<String>,
    pub(crate) ranges: Vec<(usize, usize)>,
}

/// 解码产物（只在本窗合并期用的大件，投影矩阵约 8-15 MB/窗），段末即释放。
pub(crate) struct WinBig {
    pub(crate) flat: Vec<FlatMorpheme>,
    pub(crate) tgt_flags: Vec<bool>,
    pub(crate) dep: HashMap<String, (Vec<i32>, Vec<usize>)>,
    pub(crate) cohesion_proj: Option<RelationWiseProjection>,
    pub(crate) discourse_proj: Option<SelectionProjection>,
    pub(crate) cohesion_special_indices: Vec<usize>,
    pub(crate) sent_of_morpheme: Vec<usize>,
}
