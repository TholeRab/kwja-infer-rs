//! word 阶段推理编排（阶段 7）：Juman++ 文本 → 13 头 → KNP 文本。
//!
//! 流程照抄 `WordModuleWriter` 扁平链（`word_module_writer.py`）：
//! 1. 解析 → 分窗（阶段 6 的 `split_flat_word_document`）
//! 2. 每窗：编码（special 追加＋pad）→ ONNX encoder → FIRST 池化 → 13 头
//! 3. 解码：读音（token 级 argmax → 词级归并）、四路属性（掩码 argmax）、
//!    NE（CRF Viterbi）、词素/基本句素性（多标签 sigmoid 阈值）、
//!    依存（topk＋类型 argmax＋成环回退）、cohesion（掩码 gather argmax）、
//!    談話関係（softmax 阈值）
//! 4. 组装：切分（chunk_plain）→ 三小步（只作用目标句）→ 合并体视图
//!    （缓冲句用**首次那窗**的结构，坐标平移对齐）→ cohesion/discourse
//! 5. 缓冲：首窗定稿（first-window-wins），按插入序渲染 KNP

use std::collections::{HashMap, HashSet};
use std::path::Path;

use kwja_encoder::{Encoder, WordEncoderInput};
use kwja_heads::forward::{RelationWiseProjection, SelectionProjection, WordHeads};
use kwja_heads::sequence_labeling::{
    get_morpheme_attribute_predictions, viterbi_decode, CrfParameters,
};
use kwja_tokenizer::word_side::WordTokenizer;

use crate::chunking::{add_bp_features, add_ne, chunk_plain, BasePhrase, Chunk, Phrase};
use crate::cohesion::{add_cohesion, CohesionDoc, RelRow};
use crate::dependency::{add_dependency, DepCandidate};
use crate::discourse::add_discourse;
use crate::jinf::Jinf;
use crate::jumandic::JumanDic;
use crate::knp_emit::{bp_header, feature_string, morpheme_line};
use crate::morpheme_build::{
    build_morphemes, FeatureValue, FlatMorpheme, MorphemeBuildInput, OrderedFeats,
};
use crate::word_stage::{parse_jumanpp_document, split_flat_word_document, WordWindow};
use crate::word_tables::WordTables;

/// word 阶段推理器。编码器按调用传入（与 char_stage 同风格）。
pub struct WordPredictor {
    heads: WordHeads,
    tables: WordTables,
    jumandic: JumanDic,
    jinf: Jinf,
    tokenizer: WordTokenizer,
    max_seq_length: usize,
    max_token_length: usize,
    stride: i64,
    crf: CrfParameters,
}

impl WordPredictor {
    /// `assets_dir`：本仓 `assets/`（models/word_encoder7.onnx 由调用方加载，
    /// 这里读 models/word_heads.safetensors、models/word_tables.json、
    /// tokenizers/…、dicts/…）。
    pub fn new(assets_dir: &Path, max_seq_length: usize, stride: i64) -> Result<Self, String> {
        let tables = WordTables::load(&assets_dir.join("models/word_tables.json"))?;
        let heads = WordHeads::load(&assets_dir.join("models/word_heads.safetensors"))?;
        heads.validate()?;
        let jumandic = JumanDic::load(&assets_dir.join("dicts/jumandic"))?;
        let jinf = Jinf::load(&assets_dir.join("dicts"))?;
        // tokenizer：8 个 additional specials（含 " "）＋ 数据集的 7 个
        let additional: Vec<&str> = tables.special_tokens.iter().map(|s| s.as_str()).collect();
        let dataset_specials = tables.dataset_special_tokens();
        let tokenizer = WordTokenizer::load(
            tokenizer_json(assets_dir)?,
            &additional,
            &dataset_specials,
            max_seq_length,
        )?;
        // max_token_length = max_seq_length - len(extra_special_tokens) - 2
        //（word_inference.py:115；extra = 8 个 additional specials）
        let max_token_length = max_seq_length - additional.len() - 2;
        let num_tags = heads.num_ne_tags;
        let crf = CrfParameters {
            start_transitions: heads.crf_start.clone(),
            transitions: (0..num_tags)
                .map(|i| heads.crf_transitions[i * num_tags..(i + 1) * num_tags].to_vec())
                .collect(),
            end_transitions: heads.crf_end.clone(),
            o_index: 0, // NE_TAGS[0] == "O"
        };
        Ok(Self {
            heads,
            tables,
            jumandic,
            jinf,
            tokenizer,
            max_seq_length,
            max_token_length,
            stride,
            crf,
        })
    }

    /// 一份 char 产出（Juman++ 文本）→ KNP 文本（对应一次 predict 批）。
    pub fn predict_text(
        &mut self,
        encoder: &mut dyn Encoder,
        juman_text: &str,
    ) -> Result<String, String> {
        // ★doc_id 不能留空串：Python 侧分窗之前先跑 add_doc_ids_flat
        //   （datasets/utils.py:24-30），把空 doc_id 补成 "<时间戳>-<序号>"；
        //   分窗时 to_sub_doc_id 再把子文档拼成 "<前缀>-s{stride}i{idx}"。
        //   留空串会生成 "-s21i0" 这类**匹配不上** SUB_DOC_PAT 的 id
        //   （正则要求前缀至少一个非 '-' 字符），于是 target_sentence_flags
        //   走「不匹配 ⇒ 全 True」分支——整窗 21 句全被当成目标句，
        //   buffer 与 cohesion 的 targets 全错（实测第 1 章 rel 差 66 处）。
        //   前缀取什么值不影响输出：输出前 strip_subdoc 会把它剥掉。
        let document = parse_jumanpp_document(juman_text, "0");
        let windows = split_flat_word_document(
            &document,
            self.max_token_length,
            &self.tokenizer,
            self.stride,
        )?;

        let mut buf = Buffer::default();
        for window in &windows {
            if let Some(out) = self.process_window(encoder, window)? {
                buf.absorb(out, &self.tables);
            }
        }
        Ok(buf.render(&self.tables))
    }

    /// 一个窗口：编码 → 前向 → 解码 → 组装。超长窗返回 `None`
    ///（上游 warning 后 `continue`——本窗不产出任何内容）。
    #[allow(clippy::too_many_lines)]
    fn process_window(
        &mut self,
        encoder: &mut dyn Encoder,
        window: &WordWindow,
    ) -> Result<Option<WindowOut>, String> {
        let surfs = window.surfs();
        let lemmas = window.lemmas();
        let n_morphemes = surfs.len();
        let encoding = self.tokenizer.encode_window(&surfs)?;
        let n_specials = self.tables.dataset_special_tokens().len();
        // 超长窗：word_inference.py:118-121 warning + continue
        if encoding.num_tokens > self.max_seq_length - n_specials {
            return Ok(None);
        }

        // ---- ONNX encoder → LAST_HIDDEN ----
        let special_token_indices: Vec<i64> = (0..n_specials as i64)
            .map(|i| encoding.num_tokens as i64 + i)
            .collect();
        let hidden = encoder
            .run_word(&WordEncoderInput {
                input_ids: encoding.input_ids.clone(),
                attention_mask: encoding.attention_mask.clone(),
                special_token_indices,
                batch: 1,
                seq: self.max_seq_length,
            })
            .map_err(|e| e.to_string())?;
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

        // ---- 读音：token 级隐状态 → argmax → 词级归并（word.py:148）----
        let reading_logits = self.heads.reading.logits(&hidden_rows);
        let reading_names: Vec<&str> = reading_logits
            .iter()
            .map(|row| self.tables.id2reading[argmax(row)].as_str())
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

        // ---- 四路属性（掩码 argmax）----
        let pos_logits = self.heads.pos.logits(&pooled);
        let subpos_logits = self.heads.subpos.logits(&pooled);
        let conjtype_logits = self.heads.conjtype.logits(&pooled);
        let conjform_logits = self.heads.conjform.logits(&pooled);
        let attrs = get_morpheme_attribute_predictions(
            &pos_logits[..n_morphemes],
            &subpos_logits[..n_morphemes],
            &conjtype_logits[..n_morphemes],
            &conjform_logits[..n_morphemes],
        );

        // ---- 词素素性 / NE / 基本句素性 ----
        let wf_prob = self.heads.word_feature.probabilities(&pooled);
        let ne_logits = self.heads.ne.logits(&pooled);
        let bf_prob = self.heads.base_phrase_feature.probabilities(&pooled);

        // ---- 组装词素（词典/jinf/同形词；readings 即模型读音，同时用于
        //      意味素性查找与 KNP 读方字段）----
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
        let sids: Vec<String> = window.sentences.iter().map(|s| strip_subdoc(&s.sid)).collect();

        // ---- 切分 ----
        let mut chunks = chunk_plain(&mut flat, &ranges, &wf_prob, &self.tables.word_features);

        // ---- NE（只作用目标句）----
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
            // NE 掩码 = 目标句词素
            let mut ne_mask = vec![false; self.max_seq_length];
            ne_mask[lo..hi].fill(true);
            let preds = viterbi_decode(std::slice::from_ref(&ne_logits), &[ne_mask], &self.crf);
            let rows = add_ne(&flat, &bp_of_ms, &preds[0], lo, hi, &self.tables.ne_tags);
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

        // ---- 基本句素性＋三小步之 節-区切/節-主辞（只作用目标句）----
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

        // ---- 依存（只作用目标句；topk＋类型＋成环回退）----
        // ★[ROOT] 在 special_tokens 里的位置是**第 6 个**（[著者] [読者] [不特定:人]
        //   [不特定:物] [NULL] [NA] [ROOT]），不是第 0 个。Python 侧取的是
        //   special_token_indexer.get_morpheme_level_index("[ROOT]")（word_inference.py:180），
        //   即 num_morphemes + 该下标。写成 num_morphemes 会让依存掩码把根画到
        //   [著者] 列上，整章 parent 全错（实测第 1 章 72 处基本句行、8 处文节行）。
        let root_offset = self
            .tables
            .dataset_special_tokens()
            .iter()
            .position(|t| *t == "[ROOT]")
            .expect("special_tokens 里没有 [ROOT]");
        let root_gi = n_morphemes + root_offset;
        let dep_proj = self.heads.dependency.project(&pooled);
        let mut dep: HashMap<String, (Vec<i32>, Vec<usize>)> = HashMap::new();
        for (si, tgt) in tgt_flags.iter().enumerate() {
            if !tgt {
                continue;
            }
            let bps: Vec<&BasePhrase> =
                chunks[si].phrases.iter().flat_map(|p| &p.bps).collect();
            let head_gis: Vec<usize> = bps.iter().map(|b| b.head_gi).collect();
            // 每个主辞行：掩码 → topk 候选 → 类型 argmax
            let mut dep_preds: Vec<Vec<DepCandidate>> = vec![Vec::new(); self.max_seq_length];
            let mask = dependency_mask(&ranges, si, root_gi, self.max_seq_length);
            for &hg in &head_gis {
                let row = dep_proj.row(hg);
                let mut masked: Vec<f32> = Vec::with_capacity(self.max_seq_length);
                let row_mask = &mask[hg * self.max_seq_length..(hg + 1) * self.max_seq_length];
                for (j, &ok) in row_mask.iter().enumerate() {
                    masked.push(if ok { row[j][0] } else { kwja_heads::mask_tables::MASKED });
                }
                for parent_gi in topk_indices(&masked, self.tables.dependency_topk) {
                    let dtype =
                        self.heads.dependency_type.predict_pair(&pooled[hg], &pooled[parent_gi]);
                    dep_preds[hg].push(DepCandidate { parent_gi, dep_type_idx: dtype });
                }
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

        // ---- cohesion / discourse 的投影（解码在缓冲合并体上做）----
        let cohesion_proj = self.heads.cohesion.project(&pooled);
        let discourse_proj = self.heads.discourse.project(&pooled);

        // 词素 → 句序（cohesion 同句判据）
        let mut sent_of_morpheme = vec![0usize; self.max_seq_length];
        for (si, (lo, hi)) in ranges.iter().enumerate() {
            sent_of_morpheme[*lo..*hi].fill(si);
        }

        Ok(Some(WindowOut {
            flat,
            chunks,
            sids,
            ranges,
            tgt_flags,
            dep,
            cohesion_proj: Some(cohesion_proj),
            discourse_proj: Some(discourse_proj),
            // ★cohesion 的 special 候选列不含 [ROOT]：Python 侧取
            //   get_morpheme_level_indices(only_cohesion=True)，把 [ROOT] 滤掉了
            //   （[ROOT] 是依存树的虚拟根，不参与指代消解）。多给一列会让 argmax
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
        }))
    }
}

/// 读音归并（`utils/reading_prediction.py` 的 get_word_level_readings）。
fn get_word_level_readings(
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

/// 依存掩码：目标句的块对角（除对角线）＋ [ROOT] 列（扁平 seq×seq）。
fn dependency_mask(ranges: &[(usize, usize)], si: usize, root_gi: usize, seq: usize) -> Vec<bool> {
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

/// torch.topk 的等价物：降序取 k 个下标；并列时取先出现者。
fn topk_indices(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    idx.truncate(k);
    idx
}

/// `-s\d+i\d+-` → `-`（writer 的 _strip_doc；不能剥 `2026xxxxxxxx-N-` 前缀）。
fn strip_subdoc(sid: &str) -> String {
    static PAT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pat = PAT.get_or_init(|| {
        regex::Regex::new(r"-s\d+i\d+-").expect("子文档标记正则编译失败")
    });
    pat.replace_all(sid, "-").into_owned()
}

fn argmax(row: &[f32]) -> usize {
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

/// 窗口处理产物（缓冲吸收后仅留渲染所需字段）。
struct WindowOut {
    flat: Vec<FlatMorpheme>,
    chunks: Vec<Chunk>,
    sids: Vec<String>,
    ranges: Vec<(usize, usize)>,
    tgt_flags: Vec<bool>,
    dep: HashMap<String, (Vec<i32>, Vec<usize>)>,
    cohesion_proj: Option<RelationWiseProjection>,
    discourse_proj: Option<SelectionProjection>,
    cohesion_special_indices: Vec<usize>,
    sent_of_morpheme: Vec<usize>,
}

/// 合并体里的基本句视图（renumber 的扁平版）。
struct DocBp {
    sid: String,
    index: usize,
    head_gidx: usize,
    head_text: String,
    feats: OrderedFeats,
    win_ms0: usize,
}

/// 留档的窗口数据（渲染用）。
struct StoredWin {
    flat: Vec<FlatMorpheme>,
    chunks: Vec<Chunk>,
    sids: Vec<String>,
    ranges: Vec<(usize, usize)>,
    tgt_flags: Vec<bool>,
}

/// 输出缓冲：首窗定稿、按插入序渲染。
#[derive(Default)]
struct Buffer {
    wins: Vec<StoredWin>,
    order: Vec<String>,
    owner: HashMap<String, usize>,
    dep_rows: HashMap<String, (Vec<i32>, Vec<usize>)>,
    rel_rows: HashMap<(String, usize), Vec<RelRow>>,
    dis_rows: HashMap<(String, usize), String>,
}

impl Buffer {
    fn absorb(&mut self, out: WindowOut, tables: &WordTables) {
        let win = self.wins.len();
        // 1) 先入册（cohesion 的合并体要用「上一窗结束时」的 owner 状态）
        self.wins.push(StoredWin {
            flat: out.flat,
            chunks: out.chunks,
            sids: out.sids,
            ranges: out.ranges,
            tgt_flags: out.tgt_flags,
        });
        let stored = &self.wins[win];

        // 2) 合并体视图：缓冲句用首次那窗的结构（坐标平移对齐），否则本窗的
        //（统一持有克隆，规避生长中的借用）
        let mut views: Vec<(usize, Chunk)> = Vec::with_capacity(stored.sids.len());
        for (si, sid) in stored.sids.iter().enumerate() {
            if let Some(&ow) = self.owner.get(sid)
                && ow != win
            {
                let oi = self.wins[ow]
                    .sids
                    .iter()
                    .position(|s| s == sid)
                    .expect("缓冲句不在原窗");
                let shift = stored.ranges[si].0 as i64 - self.wins[ow].ranges[oi].0 as i64;
                views.push((si, shift_chunk(&self.wins[ow].chunks[oi], shift)));
                continue;
            }
            views.push((si, stored.chunks[si].clone()));
        }

        // 3) renumber：基本句扁平列表（句内序号、head、素性、win_ms0）
        let mut doc_bps: Vec<DocBp> = Vec::new();
        for (si, ch) in &views {
            let si = *si;
            for (index, bp) in ch.phrases.iter().flat_map(|p| &p.bps).enumerate() {
                doc_bps.push(DocBp {
                    sid: stored.sids[si].clone(),
                    index,
                    head_gidx: bp.head_gi,
                    head_text: bp.head_text.clone(),
                    feats: bp.feats.clone(),
                    win_ms0: bp.first_ms(),
                });
            }
        }

        // 4) cohesion（目标 = 本窗目标句的基本句；候选 = 合并体全部基本句＋special）
        let tgt_sids: HashSet<&str> = stored
            .sids
            .iter()
            .enumerate()
            .filter(|(si, _)| stored.tgt_flags[*si])
            .map(|(_, s)| s.as_str())
            .collect();
        let targets: HashSet<(String, usize)> = doc_bps
            .iter()
            .filter(|b| tgt_sids.contains(b.sid.as_str()))
            .map(|b| (b.sid.clone(), b.index))
            .collect();
        // 同名键先清（上游每窗对 targets 先 pop 再 append）
        for key in &targets {
            self.rel_rows.remove(key);
        }
        {
            let cohesion_proj = out.cohesion_proj.expect("cohesion 投影缺失");
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
            let sent_of_morpheme = &out.sent_of_morpheme;
            let rows = add_cohesion(
                &doc,
                &|gi: usize, cols: &[usize]| cohesion_proj.row_gathered(gi, cols),
                &tables.cohesion_tasks,
                &|t: &str| tables.cohesion_task2name(t),
                &|t: &str| tables.cohesion_task2rels(t),
                &|a: usize, c: usize| sent_of_morpheme[a] == sent_of_morpheme[c],
                &out.cohesion_special_indices,
                &tables.dataset_special_tokens().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                &tables.exophora_referents,
                tables.restrict_cohesion_target,
                tables.cohesion_rels().len(),
            );
            for (bi, row) in rows {
                let key = (doc_bps[bi].sid.clone(), doc_bps[bi].index);
                if targets.contains(&key) {
                    self.rel_rows.entry(key).or_default().push(row);
                }
            }
        }

        // 5) discourse（节 = 带 節-区切 的基本句；行列 = 节首词素）
        {
            let discourse_proj = out.discourse_proj.expect("discourse 投影缺失");
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
                    self.dis_rows.insert(key, s);
                }
            }
        }

        // 6) 依存结果首窗定稿
        for (sid, d) in out.dep {
            self.dep_rows.entry(sid).or_insert(d);
        }

        // 7) 缓冲登记（首窗定稿）
        for (si, sid) in stored.sids.iter().enumerate() {
            if stored.tgt_flags[si] && !self.owner.contains_key(sid) {
                self.order.push(sid.clone());
                self.owner.insert(sid.clone(), win);
            }
        }
    }

    /// 渲染整份 KNP 文本（`_flush` ＋ `_flat_lines`）。
    fn render(&self, tables: &WordTables) -> String {
        let mut lines: Vec<String> = Vec::new();
        for sid in &self.order {
            let w = &self.wins[self.owner[sid]];
            let oi = w.sids.iter().position(|s| s == sid).expect("缓冲句不在原窗");
            let ch = &w.chunks[oi];
            let (parents, dtypes) = self
                .dep_rows
                .get(sid)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            lines.push(format!("# S-ID:{sid} kwja:2.6.0"));

            // 文节层 parent：边扫边覆盖（上游 _flat_lines 的语义）
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

/// tokenizer.json 定位（HF snapshot 目录唯一）。
fn tokenizer_json(assets_dir: &Path) -> Result<std::path::PathBuf, String> {
    let snapshots = assets_dir
        .join("tokenizers")
        .join("models--ku-nlp--deberta-v2-base-japanese")
        .join("snapshots");
    let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshots)
        .map_err(|e| format!("tokenizer snapshots 目录不存在：{e}"))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    hits.sort();
    if hits.len() != 1 {
        return Err(format!("tokenizer snapshot 不唯一：{hits:?}"));
    }
    Ok(hits.remove(0).join("tokenizer.json"))
}
