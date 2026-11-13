//! word 阶段合并与渲染：单窗合并、首窗定稿缓冲与按插入序的 KNP 渲染。

use std::collections::{HashMap, HashSet};

use crate::chunking::{BasePhrase, Chunk, Phrase};
use crate::cohesion::{CohesionDoc, RelRow, add_cohesion};
use crate::discourse::add_discourse;
use crate::knp_emit::{bp_header, feature_string, morpheme_line};
use crate::morpheme_build::{FeatureValue, FlatMorpheme, OrderedFeats};
use crate::word_decode::{WinBig, WinKeep};
use crate::word_tables::WordTables;

/// 合并体里的基本句视图（扁平化的重新编号）。
struct DocBp {
    sid: String,
    index: usize,
    head_gidx: usize,
    head_text: String,
    feats: OrderedFeats,
    win_ms0: usize,
}

/// 留档的窗口数据（渲染用；合并期的 ranges/tgt_flags 不进留档）。
struct StoredWin {
    flat: Vec<FlatMorpheme>,
    chunks: Vec<Chunk>,
    sids: Vec<String>,
}

/// 输出缓冲：首窗定稿、按插入序渲染。
#[derive(Default)]
pub(crate) struct Buffer {
    wins: Vec<StoredWin>,
    pub(crate) order: Vec<String>,
    pub(crate) owner: HashMap<String, usize>,
    dep_rows: HashMap<String, (Vec<i32>, Vec<usize>)>,
    rel_rows: HashMap<(String, usize), Vec<RelRow>>,
    dis_rows: HashMap<(String, usize), String>,
}

/// 一次窗合并的产物：渲染要用的本窗留档与三类行。
/// 由 absorb_one 按窗并行产出，Buffer::fold 按窗序写入。
pub(crate) struct AbsorbedWin {
    win: StoredWin,
    rel_rows: HashMap<(String, usize), Vec<RelRow>>,
    dis_rows: HashMap<(String, usize), String>,
    dep_rows: HashMap<String, (Vec<i32>, Vec<usize>)>,
}

/// owner/order 登记：sid 的 owner 是它首次作为目标句出现的窗
/// （非跳过窗序号，与渲染时的 wins 下标一致）。
/// 按窗序逐窗调用；目标句集合按窗不重叠，首窗定稿语义不变。
pub(crate) fn register_owner(
    win: usize,
    sids: &[String],
    tgt_flags: &[bool],
    owner: &mut HashMap<String, usize>,
    order: &mut Vec<String>,
) {
    for (si, sid) in sids.iter().enumerate() {
        if tgt_flags[si] && !owner.contains_key(sid) {
            order.push(sid.clone());
            owner.insert(sid.clone(), win);
        }
    }
}

/// 单窗合并。
///
/// 只读三样：本窗解码产物（keep 加 big）、已登记的 owner、之前各窗的
///   小数据 live（合并体视图里缓冲句的 chunk 来源）。不写任何累积状态，
///   窗与窗可并行。产出的行由 Buffer::fold 按窗序并入；目标键按窗不重叠，
///   并入顺序不影响结果。
pub(crate) fn absorb_one(
    win: usize,
    keep: &WinKeep,
    big: &WinBig,
    live: &[WinKeep],
    owner: &HashMap<String, usize>,
    tables: &WordTables,
) -> AbsorbedWin {
    // 2) 合并体视图：缓冲句用首次那窗的结构（坐标平移对齐），否则本窗的
    //（统一持有克隆，规避生长中的借用）
    let mut views: Vec<(usize, Chunk)> = Vec::with_capacity(keep.sids.len());
    for (si, sid) in keep.sids.iter().enumerate() {
        if let Some(&ow) = owner.get(sid)
            && ow != win
        {
            let oi = live[ow]
                .sids
                .iter()
                .position(|s| s == sid)
                .expect("缓冲句不在原窗");
            let shift = keep.ranges[si].0 as i64 - live[ow].ranges[oi].0 as i64;
            views.push((si, shift_chunk(&live[ow].chunks[oi], shift)));
            continue;
        }
        views.push((si, keep.chunks[si].clone()));
    }

    // 3) renumber：基本句扁平列表（句内序号、head、素性、win_ms0）
    let mut doc_bps: Vec<DocBp> = Vec::new();
    for (si, ch) in &views {
        let si = *si;
        for (index, bp) in ch.phrases.iter().flat_map(|p| &p.bps).enumerate() {
            doc_bps.push(DocBp {
                sid: keep.sids[si].clone(),
                index,
                head_gidx: bp.head_gi,
                head_text: bp.head_text.clone(),
                feats: bp.feats.clone(),
                win_ms0: bp.first_ms(),
            });
        }
    }

    // 4) cohesion（目标 = 本窗目标句的基本句；候选 = 合并体全部基本句+special）
    let tgt_sids: HashSet<&str> = keep
        .sids
        .iter()
        .enumerate()
        .filter(|(si, _)| big.tgt_flags[*si])
        .map(|(_, s)| s.as_str())
        .collect();
    let targets: HashSet<(String, usize)> = doc_bps
        .iter()
        .filter(|b| tgt_sids.contains(b.sid.as_str()))
        .map(|b| (b.sid.clone(), b.index))
        .collect();
    // 本窗目标键此前不存在于缓冲（键按窗唯一），先清再加由 fold 统一处理
    let mut rel_rows: HashMap<(String, usize), Vec<RelRow>> = HashMap::new();
    {
        let cohesion_proj = big.cohesion_proj.as_ref().expect("cohesion 投影缺失");
        let bp_head_gi: Vec<usize> = doc_bps.iter().map(|b| b.head_gidx).collect();
        let bp_index: Vec<usize> = doc_bps.iter().map(|b| b.index).collect();
        let bp_sid: Vec<String> = doc_bps.iter().map(|b| b.sid.clone()).collect();
        let bp_head_text: Vec<String> = doc_bps.iter().map(|b| b.head_text.clone()).collect();
        let bp_feats: Vec<OrderedFeats> = doc_bps.iter().map(|b| b.feats.clone()).collect();
        let doc = CohesionDoc {
            bp_head_gi: &bp_head_gi,
            bp_index: &bp_index,
            bp_sid: &bp_sid,
            bp_head_text: &bp_head_text,
            bp_feats: &bp_feats,
        };
        let sent_of_morpheme = &big.sent_of_morpheme;
        // 只给「目标句的基本句」算 cohesion 行——合并体里其余基本句的行
        //   在过滤侧本来就被 targets 过滤丢弃，先不算才是纯省计算，结果不变。
        let wanted: Vec<bool> = doc_bps
            .iter()
            .map(|b| targets.contains(&(b.sid.clone(), b.index)))
            .collect();
        let rows = add_cohesion(
            &doc,
            &wanted,
            &|gi: usize, cols: &[usize]| cohesion_proj.row_gathered(gi, cols),
            &tables.cohesion_tasks,
            &|t: &str| tables.cohesion_task2name(t),
            &|t: &str| tables.cohesion_task2rels(t),
            &|a: usize, c: usize| sent_of_morpheme[a] == sent_of_morpheme[c],
            &big.cohesion_special_indices,
            &tables
                .dataset_special_tokens()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            &tables.exophora_referents,
            tables.restrict_cohesion_target,
            tables.cohesion_rels().len(),
        );
        for (bi, row) in rows {
            let key = (doc_bps[bi].sid.clone(), doc_bps[bi].index);
            if targets.contains(&key) {
                rel_rows.entry(key).or_default().push(row);
            }
        }
    }

    // 5) discourse（节 = 带 節-区切 的基本句；行列 = 节首词素）
    let mut dis_rows: HashMap<(String, usize), String> = HashMap::new();
    {
        let discourse_proj = big.discourse_proj.as_ref().expect("discourse 投影缺失");
        let cls: Vec<&DocBp> = doc_bps
            .iter()
            .filter(|b| b.feats.iter().any(|(k, _)| k.starts_with("節-区切")))
            .collect();
        let rows = add_discourse(
            &cls.iter().map(|b| b.win_ms0).collect::<Vec<_>>(),
            &cls.iter().map(|b| b.index).collect::<Vec<_>>(),
            &cls.iter().map(|b| b.sid.clone()).collect::<Vec<_>>(),
            &|i: usize, j: usize| discourse_proj.row_gathered(i, &[j]).remove(0),
            &tables.discourse_relations,
            tables.discourse_threshold,
        );
        for (ci, s) in rows {
            let key = (cls[ci].sid.clone(), cls[ci].index);
            if targets.contains(&key) {
                dis_rows.insert(key, s);
            }
        }
    }

    // 6) 依存结果（键 = 目标句 sid，按窗唯一；「首窗定稿」语义在 fold 里保留）
    let dep_rows: HashMap<String, (Vec<i32>, Vec<usize>)> = big.dep.clone();

    AbsorbedWin {
        win: StoredWin {
            flat: big.flat.clone(),
            chunks: keep.chunks.clone(),
            sids: keep.sids.clone(),
        },
        rel_rows,
        dis_rows,
        dep_rows,
    }
}

impl Buffer {
    /// 按窗序把 absorb_one 的产物并入累积缓冲。
    /// 目标键按窗不重叠，rel/dis 用 insert 即可；
    /// dep 保留 or_insert 的首窗定稿语义。wins 必须按窗序 push
    /// （渲染按 owner 下标取窗）。
    pub(crate) fn fold(&mut self, a: AbsorbedWin) {
        for (key, rows) in a.rel_rows {
            self.rel_rows.insert(key, rows);
        }
        for (key, s) in a.dis_rows {
            self.dis_rows.insert(key, s);
        }
        for (sid, d) in a.dep_rows {
            self.dep_rows.entry(sid).or_insert(d);
        }
        self.wins.push(a.win);
    }

    /// 渲染整份 KNP 文本。
    pub(crate) fn render(&self, tables: &WordTables) -> String {
        let mut lines: Vec<String> = Vec::new();
        for sid in &self.order {
            let w = &self.wins[self.owner[sid]];
            let oi = w
                .sids
                .iter()
                .position(|s| s == sid)
                .expect("缓冲句不在原窗");
            let ch = &w.chunks[oi];
            let (parents, dtypes) = self
                .dep_rows
                .get(sid)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            lines.push(format!("# S-ID:{sid} kwja:2.6.0"));

            // 文节层 parent：边扫边覆盖
            let mut ph_of_bp: Vec<usize> = Vec::new();
            for (pj, p) in ch.phrases.iter().enumerate() {
                for _ in &p.bps {
                    ph_of_bp.push(pj);
                }
            }
            let mut ph_par = vec![None; ch.phrases.len()];
            let mut ph_dt = vec![None; ch.phrases.len()];
            for (bi, &par) in parents.iter().enumerate() {
                let pj = ph_of_bp[bi];
                if par == -1 {
                    ph_par[pj] = Some(-1i32);
                    ph_dt[pj] = Some("D".to_string());
                } else if ph_of_bp[par as usize] != pj {
                    ph_par[pj] = Some(ph_of_bp[par as usize] as i32);
                    ph_dt[pj] = Some(tables.dependency_types[dtypes[bi]].clone());
                }
            }

            let mut pi = 0usize;
            for (pj, p) in ch.phrases.iter().enumerate() {
                let par = ph_par[pj].unwrap_or(-1);
                let dt = ph_dt[pj].clone().unwrap_or_else(|| "D".to_string());
                lines.push(format!("* {par}{dt}"));
                for bp in &p.bps {
                    let key = (sid.clone(), pi);
                    let mut feats = bp.feats.clone();
                    if let Some(dsc) = self.dis_rows.get(&key) {
                        feats.set("談話関係", FeatureValue::Text(dsc.clone()));
                    }
                    let rels: Vec<(String, String, Option<String>, Option<usize>)> = self
                        .rel_rows
                        .get(&key)
                        .map(|rows| {
                            rows.iter()
                                .map(|r| {
                                    (r.rel.clone(), r.target.clone(), r.sid.clone(), r.bp_index)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    lines.push(bp_header(
                        parents.get(pi).copied().unwrap_or(0),
                        &feature_string(&feats),
                        &rels,
                        Some(&tables.dependency_types[dtypes.get(pi).copied().unwrap_or(0)]),
                    ));
                    for &mi in &bp.ms {
                        lines.push(morpheme_line(&w.flat[mi]));
                    }
                    pi += 1;
                }
            }
            lines.push("EOS".to_string());
        }
        if lines.is_empty() {
            String::new()
        } else {
            lines.join("\n") + "\n"
        }
    }
}

/// 把旧窗的 chunk 平移到本窗坐标（ms/head_gi 加 shift）。
fn shift_chunk(ch: &Chunk, shift: i64) -> Chunk {
    Chunk {
        phrases: ch
            .phrases
            .iter()
            .map(|p| Phrase {
                bps: p
                    .bps
                    .iter()
                    .map(|b| BasePhrase {
                        ms: b.ms.iter().map(|&m| (m as i64 + shift) as usize).collect(),
                        head_gi: (b.head_gi as i64 + shift) as usize,
                        head_text: b.head_text.clone(),
                        feats: b.feats.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}
