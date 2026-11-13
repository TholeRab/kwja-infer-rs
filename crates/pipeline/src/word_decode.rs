//! word 阶段解码段：13 头解码一个窗，及其纯函数工具（读音归并、依存掩码与 topk、argmax）。

use rayon::prelude::*;
use std::collections::HashMap;

use kwja_heads::forward::{RelationWiseProjection, SelectionProjection};
use kwja_heads::sequence_labeling::{get_morpheme_attribute_predictions, viterbi_decode};

use crate::chunking::{BasePhrase, Chunk, add_bp_features, add_ne, chunk_plain};
use crate::dependency::{DepCandidate, add_dependency};
use crate::morpheme_build::{FeatureValue, FlatMorpheme, MorphemeBuildInput, build_morphemes};
use crate::word_encode::WindowPrep;
use crate::word_predict::WordPredictor;
use crate::word_stage::WordWindow;

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
pub(crate) fn dependency_mask(
    ranges: &[(usize, usize)],
    si: usize,
    root_gi: usize,
    seq: usize,
) -> Vec<bool> {
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
    let pat = PAT.get_or_init(|| regex::Regex::new(r"-s\d+i\d+-").expect("子文档标记正则编译失败"));
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

impl WordPredictor {
    /// 解码一个窗：13 头 + 切分 + NE + 依存候选。
    ///
    /// &self，不写任何共享状态，所以窗与窗之间可以铺满所有核。
    ///   原先这段与编码绑在同一个 &mut self 函数里，一次只能跑一个窗，
    ///   并行度上不去。
    pub(crate) fn decode_window(
        &self,
        window: &WordWindow,
        prep: WindowPrep,
    ) -> Result<(WinKeep, WinBig), String> {
        let WindowPrep { encoding, hidden } = prep;
        let surfs = window.surfs();
        let lemmas = window.lemmas();
        let n_morphemes = surfs.len();
        let dbg4 = std::env::var("KWJA_TIMING").is_ok();
        let t_enc = std::time::Instant::now();
        let shape = hidden.shape();
        let (seq_len, hid) = (shape[1], shape[2]);
        let flat_hidden: Vec<f32> = hidden.as_slice().expect("encoder 输出应连续").to_vec();
        // token 级隐状态（读音头用）与 FIRST 池化（其余头用）
        let hidden_rows: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| flat_hidden[t * hid..(t + 1) * hid].to_vec())
            .collect();
        let pooled: Vec<Vec<f32>> = (0..seq_len)
            .map(|m| {
                let first = encoding.subword_first_token(m, n_morphemes);
                flat_hidden[first * hid..(first + 1) * hid].to_vec()
            })
            .collect();

        // ---- 读音 + 其余头：一个并行组 ----
        // reading 吃 hidden_rows、其余头吃 pooled，互不依赖；
        //   reading 输出类别最多、计算最重。
        //   组内只算 logits/概率，argmax 与之后的归并留在组外串行做——
        //   那些依赖前一步结果，并行没有意义。
        let h = &self.heads;
        const T_READING: usize = 0;
        const T_POS: usize = 1;
        const T_SUBPOS: usize = 2;
        const T_CONJTYPE: usize = 3;
        const T_CONJFORM: usize = 4;
        const T_WF: usize = 5;
        const T_NE: usize = 6;
        const T_BF: usize = 7;
        // 这些头下游只读词素行——
        //   pos/subpos/conjtype/conjform 下游显式取 [..n_morphemes]，
        //   wf 的 chunk_plain、bf 的 add_bp_features 也只按词素/主辞下标取行，
        //   reading 只被 rmap 里真词素的 token 位读（填充与 special 位从不读）。
        //   forward_rows 是逐行 matvec、行与行独立，只算这些行结果逐字节相同。
        //   NE 例外：它走全序列 Viterbi（掩码路径跨整窗），保持原样。
        let token_rows = &hidden_rows[..encoding.num_tokens];
        let morph_rows = &pooled[..n_morphemes];
        // 12 个头的重前向整体切 GPU——一张融合图出全部 logits/概率与 pairwise
        // 投影。行集口径与 CPU 路径一致：reading 吃 token 级，其余头吃含 special 区
        // 的全量 pooled（依存与 cohesion 的候选列要取 [ROOT] 等特殊词位），词素级
        // 输出由下游再切 [..n_morphemes]。
        let gpu_out: Option<kwja_heads::gpu_heads::FusedHeadsOutput> = match self.gpu_heads.as_ref()
        {
            Some(g) => {
                // pooled 只送「真词素+special 区」——其后的填充行没有任何
                //   消费者（依存/cohesion/discourse 的行列下标全都 < 这里），
                //   GPU 的投影与 reading 计算量随行数走，不送白算的填充行。
                let usable = n_morphemes + self.tables.dataset_special_tokens().len();
                Some(g.forward_fused(token_rows, &pooled[..usable])?)
            }
            None => None,
        };
        let (reading_argmax, wf_prob, ne_logits, bf_prob, attrs) = match &gpu_out {
            Some(o) => {
                let attrs = get_morpheme_attribute_predictions(
                    &o.pos_logits[..n_morphemes],
                    &o.subpos_logits[..n_morphemes],
                    &o.conjtype_logits[..n_morphemes],
                    &o.conjform_logits[..n_morphemes],
                );
                (
                    o.reading_argmax.clone(),
                    o.wf_prob.clone(),
                    o.ne_logits.clone(),
                    o.bpf_prob.clone(),
                    attrs,
                )
            }
            None => {
                // 九个头的耗时不均匀，二叉 join 会把大块和小块切到一起，
                //   rayon 只能按切分树并行，CPU 利用率低。改为扁平任务表交给
                //   par_iter，由线程池自己调度。
                //   必须是 IndexedParallelIterator（用 collect 到 Vec），顺序才有保证。
                //   组内只算 logits/概率，argmax 与之后的归并留在组外串行做——
                //   那些依赖前一步结果，并行没有意义。
                let outs: Vec<Vec<Vec<f32>>> = (0..8usize)
                    .into_par_iter()
                    .map(|t| match t {
                        T_READING => h.reading.logits(token_rows),
                        T_POS => h.pos.logits(morph_rows),
                        T_SUBPOS => h.subpos.logits(morph_rows),
                        T_CONJTYPE => h.conjtype.logits(morph_rows),
                        T_CONJFORM => h.conjform.logits(morph_rows),
                        T_WF => h.word_feature.probabilities(morph_rows),
                        T_NE => h.ne.logits(&pooled),
                        T_BF => h.base_phrase_feature.probabilities(morph_rows),
                        _ => unreachable!(),
                    })
                    .collect();
                let attrs = get_morpheme_attribute_predictions(
                    &outs[T_POS][..n_morphemes],
                    &outs[T_SUBPOS][..n_morphemes],
                    &outs[T_CONJTYPE][..n_morphemes],
                    &outs[T_CONJFORM][..n_morphemes],
                );
                (
                    outs[T_READING]
                        .iter()
                        .map(|row| argmax(row))
                        .collect::<Vec<usize>>(),
                    outs[T_WF].clone(),
                    outs[T_NE].clone(),
                    outs[T_BF].clone(),
                    attrs,
                )
            }
        };

        let reading_names: Vec<&str> = reading_argmax
            .iter()
            .map(|&ai| self.tables.id2reading[ai].as_str())
            .collect();
        let tokens: Vec<String> = encoding
            .input_ids
            .iter()
            .map(|&id| self.tokenizer.decode_one(id))
            .collect();
        let rmap = encoding.subword_map(n_morphemes, false);
        let word_readings = get_word_level_readings(
            &reading_names,
            &tokens,
            &rmap[..n_morphemes],
            (&self.tables.unk, &self.tables.id),
        );

        // ---- 组装词素（词典/jinf/同形词；readings 即模型读音，同时用于
        //      意味素性查找与 KNP 读方字段）----
        if dbg4 {
            eprintln!(
                "      encoder+heads {:.1} ms",
                t_enc.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_build = std::time::Instant::now();
        let mut flat = build_morphemes(
            &self.jumandic,
            &self.jinf,
            &self.tables,
            &MorphemeBuildInput {
                surfs: &surfs,
                norms: &lemmas,
                readings: &word_readings,
                pos_idx: &attrs.pos,
                subpos_idx: &attrs.subpos,
                conjtype_idx: &attrs.conjtype,
                conjform_idx: &attrs.conjform,
            },
        );

        if dbg4 {
            eprintln!(
                "      build_morphemes {:.1} ms",
                t_build.elapsed().as_secs_f64() * 1000.0
            );
        }
        // ---- 句区间与目标句 ----
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(window.sentences.len());
        {
            let mut acc = 0usize;
            for s in &window.sentences {
                ranges.push((acc, acc + s.surfs.len()));
                acc += s.surfs.len();
            }
        }
        let tgt_flags =
            crate::window::target_sentence_flags(&window.doc_id, window.sentences.len());
        let sids: Vec<String> = window
            .sentences
            .iter()
            .map(|s| strip_subdoc(&s.sid))
            .collect();

        // ---- 切分 ----
        let t_chunk = std::time::Instant::now();
        let mut chunks = chunk_plain(&mut flat, &ranges, &wf_prob, &self.tables.word_features);
        if dbg4 {
            eprintln!(
                "      chunk_plain {:.1} ms",
                t_chunk.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_ne = std::time::Instant::now();

        // ---- NE（只作用目标句）----
        //
        // 掩码是整窗全部目标句，不是当前这一句：
        //   ne_mask 按 target_mask 一次性标满，Viterbi 只跑一次。
        //   若逐句重建掩码、逐句跑 Viterbi，每句都被当成窗口里唯一的目标段，
        //   一窗多个目标句时会多报实体。
        let mut ne_mask = vec![false; self.max_seq_length];
        for (si, tgt) in tgt_flags.iter().enumerate() {
            if *tgt {
                let (lo, hi) = ranges[si];
                ne_mask[lo..hi].fill(true);
            }
        }
        let ne_preds = viterbi_decode(std::slice::from_ref(&ne_logits), &[ne_mask], &self.crf);
        for (si, tgt) in tgt_flags.iter().enumerate() {
            if !tgt {
                continue;
            }
            let (lo, hi) = ranges[si];
            let mut bp_of_ms = vec![0usize; n_morphemes];
            for (bpi, bp) in chunks[si].phrases.iter().flat_map(|p| &p.bps).enumerate() {
                for &mi in &bp.ms {
                    bp_of_ms[mi] = bpi;
                }
            }
            let rows = add_ne(&flat, &bp_of_ms, &ne_preds[0], lo, hi, &self.tables.ne_tags);
            for (bpi, key, val) in rows {
                chunks[si]
                    .phrases
                    .iter_mut()
                    .flat_map(|p| &mut p.bps)
                    .nth(bpi)
                    .expect("NE 基本句下标越界")
                    .feats
                    .push(key, FeatureValue::Text(val));
            }
        }

        // ---- 基本句素性+三小步之 節-区切/節-主辞（只作用目标句）----
        for (si, tgt) in tgt_flags.iter().enumerate() {
            if !tgt {
                continue;
            }
            let head_gis: Vec<usize> = chunks[si]
                .phrases
                .iter()
                .flat_map(|p| &p.bps)
                .map(|b| b.head_gi)
                .collect();
            add_bp_features(
                &mut chunks[si],
                &bf_prob,
                &head_gis,
                &self.tables.base_phrase_features,
            );
        }

        if dbg4 {
            eprintln!(
                "      NE+bp_features {:.1} ms",
                t_ne.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_dep = std::time::Instant::now();
        // ---- 依存（只作用目标句；topk+类型+成环回退）----
        // [ROOT] 在 special_tokens 里排第 6（[著者] [読者] [不特定:人]
        //   [不特定:物] [NULL] [NA] [ROOT]），不是第 0 个：
        //   root_gi = num_morphemes + 该下标。写成 num_morphemes 会让依存掩码
        //   把根画到 [著者] 列上，整章 parent 全错。
        let root_offset = self
            .tables
            .dataset_special_tokens()
            .iter()
            .position(|t| *t == "[ROOT]")
            .expect("special_tokens 里没有 [ROOT]");
        let root_gi = n_morphemes + root_offset;
        let t_proj = std::time::Instant::now();
        // 三组投影一次装配，GPU 输出移动进来——原先逐字段 clone，
        // 每窗白拷约 20MB 的投影矩阵。
        let (dep_proj, cohesion_proj, discourse_proj) = match gpu_out {
            Some(o) => (
                kwja_heads::forward::SelectionProjection::from_projected(
                    &self.heads.dependency,
                    o.dep_hs,
                    o.dep_ht,
                ),
                kwja_heads::forward::RelationWiseProjection::from_projected(
                    &self.heads.cohesion,
                    o.coh_source,
                    o.coh_target,
                ),
                kwja_heads::forward::SelectionProjection::from_projected(
                    &self.heads.discourse,
                    o.dis_hs,
                    o.dis_ht,
                ),
            ),
            None => (
                self.heads.dependency.project(&pooled),
                self.heads.cohesion.project(&pooled),
                self.heads.discourse.project(&pooled),
            ),
        };
        let mut dep: HashMap<String, (Vec<i32>, Vec<usize>)> = HashMap::new();
        for (si, tgt) in tgt_flags.iter().enumerate() {
            if !tgt {
                continue;
            }
            let bps: Vec<&BasePhrase> = chunks[si].phrases.iter().flat_map(|p| &p.bps).collect();
            let head_gis: Vec<usize> = bps.iter().map(|b| b.head_gi).collect();
            // 每个主辞行：先掩码，再取 topk 候选，最后做类型 argmax
            let mut dep_preds: Vec<Vec<DepCandidate>> = vec![Vec::new(); self.max_seq_length];
            let mask = dependency_mask(&ranges, si, root_gi, self.max_seq_length);
            // 行与行独立，并行算（dependency 是单窗里最重的一块）。
            //   par_iter 保序，dep_preds 的下标与串行一致。
            //   topk 前的行 logits 仍是 CPU 小点积；topk 出的全部（主辞,候选）对
            //   攒成一个批次一次过 GPU（dep_type.onnx），argmax 语义不变。
            let rows: Vec<(usize, Vec<DepCandidate>)> = match self.gpu_heads.as_ref() {
                Some(g) => {
                    let topks: Vec<(usize, Vec<usize>)> = head_gis
                        .par_iter()
                        .map(|&hg| {
                            (
                                hg,
                                dependency_topk_masked(
                                    &dep_proj,
                                    hg,
                                    &mask,
                                    self.max_seq_length,
                                    self.tables.dependency_topk,
                                ),
                            )
                        })
                        .collect();
                    let pairs: Vec<(usize, usize)> = topks
                        .iter()
                        .flat_map(|(hg, ps)| ps.iter().map(move |&p| (*hg, p)))
                        .collect();
                    let dtypes = g.dep_type_argmax(&pairs, &pooled)?;
                    let mut it = dtypes.into_iter();
                    topks
                        .into_iter()
                        .map(|(hg, ps)| {
                            let cands = ps
                                .into_iter()
                                .map(|parent_gi| DepCandidate {
                                    parent_gi,
                                    dep_type_idx: it.next().expect("dep_type 输出数应与对数一致"),
                                })
                                .collect();
                            (hg, cands)
                        })
                        .collect()
                }
                None => head_gis
                    .par_iter()
                    .map(|&hg| {
                        let mut cands = Vec::new();
                        for parent_gi in dependency_topk_masked(
                            &dep_proj,
                            hg,
                            &mask,
                            self.max_seq_length,
                            self.tables.dependency_topk,
                        ) {
                            let dtype = self
                                .heads
                                .dependency_type
                                .predict_pair(&pooled[hg], &pooled[parent_gi]);
                            cands.push(DepCandidate {
                                parent_gi,
                                dep_type_idx: dtype,
                            });
                        }
                        (hg, cands)
                    })
                    .collect(),
            };
            for (hg, cands) in rows {
                dep_preds[hg] = cands;
            }
            let (parents, dtypes) = add_dependency(
                &bps,
                &dep_preds,
                &head_gis,
                root_gi,
                self.tables.dependency_types.len(),
            );
            dep.insert(sids[si].clone(), (parents, dtypes));
        }

        if dbg4 {
            eprintln!(
                "      dependency {:.1} ms",
                t_dep.elapsed().as_secs_f64() * 1000.0
            );
        }
        // ---- cohesion / discourse 的投影（已在上面与 dep_proj 一并装配）----

        // 词素下标到句序（cohesion 同句判据）
        let mut sent_of_morpheme = vec![0usize; self.max_seq_length];
        for (si, (lo, hi)) in ranges.iter().enumerate() {
            sent_of_morpheme[*lo..*hi].fill(si);
        }

        if dbg4 {
            eprintln!(
                "      projections {:.1} ms",
                t_proj.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok((
            WinKeep {
                chunks,
                sids,
                ranges,
            },
            WinBig {
                flat,
                tgt_flags,
                dep,
                cohesion_proj: Some(cohesion_proj),
                discourse_proj: Some(discourse_proj),
                // cohesion 的 special 候选列不含 [ROOT]：
                //   [ROOT] 是依存树的虚拟根，不参与指代消解。多给一列会让 argmax
                //   落到一个本不该存在的候选上，整章 rel 全错。
                cohesion_special_indices: self
                    .tables
                    .dataset_special_tokens()
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t != "[ROOT]")
                    .map(|(i, _)| n_morphemes + i)
                    .collect(),
                sent_of_morpheme,
            },
        ))
    }
}
