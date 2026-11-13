//! cohesion 解码（`flat_cohesion.py` ＋ `flat_cohesion_mask.py` 的移植）。
//!
//! 判据全部照抄 cohesion_tools 三个 extractor 的 is_target／is_candidate：
//! - 候选：Pas/Bridging = `cand < anaph` 或（`cand > anaph` 且同句）；
//!   Coreference = 只有 `cand < anaph`。
//! - 目标：Pas = 用言 ∨ 非用言格解析；Coreference = 体言；
//!   Bridging = 体言 ∧ ¬非用言格解析。
//!
//! logits 的行/列都是**窗内词素下标**：row = anaph、col = cand；
//! special 列（6 个，[ROOT] 除外）无条件对全部词素行放开。

use rayon::prelude::*;
use crate::morpheme_build::OrderedFeats;

/// `flat_cohesion._is_target`。
pub fn is_target(name: &str, feats: &OrderedFeats) -> bool {
    match name {
        "PasExtractor" => feats.contains("用言") || feats.contains("非用言格解析"),
        "CoreferenceExtractor" => matches!(feats.get("体言"), Some(crate::morpheme_build::FeatureValue::True)),
        "BridgingExtractor" => {
            matches!(feats.get("体言"), Some(crate::morpheme_build::FeatureValue::True))
                && !feats.contains("非用言格解析")
        }
        _ => true,
    }
}

/// 一条 cohesion 关系行：`(rel, target, sid, bp_index)`；
/// sid 为 None 表示指外（exophora）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelRow {
    pub rel: String,
    pub target: String,
    pub sid: Option<String>,
    pub bp_index: Option<usize>,
}

/// cohesion 解码的输入视图（对应 Python 的 `_P` / plain 文档）。
pub struct CohesionDoc<'a> {
    /// 每个基本句的 head（窗内词素下标）。
    pub bp_head_gi: &'a [usize],
    pub bp_index: &'a [usize],
    pub bp_sid: &'a [String],
    pub bp_head_text: &'a [String],
    pub bp_feats: &'a [OrderedFeats],
}

/// cohesion 行取数闭包：`(anaph 行, gather 列) → (列数 × 通道数) 的原始 logits`。
/// ★必须是 `+ Sync`：cohesion 的 101 个行彼此独立，
///   串行算只用 1 个核，而它独占整窗 57% 的时间。
pub type CohesionRowFn<'a> = &'a (dyn Fn(usize, &[usize]) -> Vec<Vec<f32>> + Sync);

/// `add_cohesion_plain`。
///
/// `row_logits`：取第 `anaph` 行、`cols` 列的原始 logits（未掩码），
/// 形状 `(cols.len(), num_rels)`——由 heads 投影的 `row_gathered` 给出。
#[allow(clippy::too_many_arguments)]
pub fn add_cohesion(
    doc: &CohesionDoc<'_>,
    row_logits: CohesionRowFn<'_>,
    cohesion_tasks: &[String],
    task2name: &dyn Fn(&str) -> &str,
    task2rels: &dyn Fn(&str) -> Vec<String>,
    same_sentence: &dyn Fn(usize, usize) -> bool,
    special_indices: &[usize],
    special_tokens: &[String],
    exophora_types: &[String],
    restrict_target: bool,
    num_rels: usize,
) -> Vec<(usize, RelRow)> {
    let b = doc.bp_head_gi.len();
    // gather 列 = 全部基本句 head ＋ special 词素下标
    let mut gather: Vec<usize> = doc.bp_head_gi.to_vec();
    gather.extend_from_slice(special_indices);

    // 每个任务一条掩码判据（闭包里按 cand/anaph 判），先按任务收集 rel 通道
    let mut rel_channels: Vec<(String, String, usize)> = Vec::new(); // (task, rel, channel)
    {
        let rels = {
            let mut v = Vec::new();
            for task in cohesion_tasks {
                for r in task2rels(task) {
                    v.push((task.clone(), r));
                }
            }
            v
        };
        let all_rels = {
            let mut v = Vec::new();
            for task in cohesion_tasks {
                v.extend(task2rels(task));
            }
            v
        };
        for (task, rel) in rels {
            let channel = all_rels.iter().position(|r| *r == rel).unwrap();
            rel_channels.push((task, rel, channel));
        }
    }
    let _ = num_rels;

    // ★阶段一：并行算全部行。
    //   行与行之间只读共享（source/target 投影结果），没有数据依赖，
    //   是这里唯一足够大的并行粒度（每行 106 列 × 9 rel × 768 维）。
    //   先前只并行「头与头之间」，cohesion 这 57% 的时间始终单线程，
    //   所以 CPU 占用只有四成。
    //
    //   ★顺序不变：用 par_iter().map().collect() 保序，结果下标与串行逐一对应。
    let all_rows: Vec<Vec<Vec<f32>>> = (0..b)
        .into_par_iter()
        .map(|bi| row_logits(doc.bp_head_gi[bi], &gather))
        .collect();

    // ★阶段二：串行解码（掩码 + argmax）。
    //   argmax 平局取首个，顺序是语义的一部分，这里必须串行。
    let mut out = Vec::new();
    for (bi, rows) in all_rows.into_iter().enumerate() {
        let head_gi = doc.bp_head_gi[bi];
        let feats = &doc.bp_feats[bi];
        for (task, rel, channel) in &rel_channels {
            if restrict_target && !is_target(task2name(task), feats) {
                continue;
            }
            // 该任务候选判据：Coreference 只有前指；Pas/Bridging 允许同句后指
            let coref = task2name(task) == "CoreferenceExtractor";
            // 掩码 + argmax（Python：mask_logits 后 row[gather_idx].argmax()，
            // np.argmax 平局取首个）。gather 的前 b 个位置是基本句 head，
            // b.. 之后是 special 词素（[ROOT] 在最前）。
            let mut best_k = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (k, logits) in rows.iter().enumerate() {
                let allowed = if k >= b {
                    // special 列：掩码只放开 only_cohesion 的 6 个（[ROOT] 被压住）
                    special_indices.iter().any(|s| s == &gather[k])
                } else {
                    let cand_gi = gather[k];
                    let anaph_gi = head_gi;
                    cand_gi < anaph_gi
                        || (!coref && cand_gi > anaph_gi && same_sentence(cand_gi, anaph_gi))
                };
                let v = if allowed { logits[*channel] } else { kwja_heads::mask_tables::MASKED };
                if v > best_v {
                    best_v = v;
                    best_k = k;
                }
            }
            if best_k < b {
                out.push((
                    bi,
                    RelRow {
                        rel: rel.clone(),
                        target: doc.bp_head_text[best_k].clone(),
                        sid: Some(doc.bp_sid[best_k].clone()),
                        bp_index: Some(doc.bp_index[best_k]),
                    },
                ));
            } else {
                let nm = special_tokens[best_k - b].trim_start_matches('[').trim_end_matches(']');
                if exophora_types.iter().any(|e| e == nm) {
                    out.push((
                        bi,
                        RelRow {
                            rel: rel.clone(),
                            target: nm.to_string(),
                            sid: None,
                            bp_index: None,
                        },
                    ));
                }
            }
        }
    }
    out
}
