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

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use kwja_encoder::{Encoder, ParallelEncoder, WordEncoderInput};
use kwja_heads::forward::{RelationWiseProjection, SelectionProjection, WordHeads};
use kwja_heads::sequence_labeling::{
    get_morpheme_attribute_predictions, viterbi_decode, CrfParameters,
};
use kwja_tokenizer::word_side::{WindowEncoding, WordTokenizer};

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
use crate::word_stage::{parse_jumanpp_documents_by_doc, split_flat_word_document, WordWindow};
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
        // ★必须按文档切分后再逐文档滑窗（parse_jumanpp_documents_by_doc，
        //   上游 word_inference.py:41-44 的 chunk_by_document 口径）。以前这里
        //   把整份输入当一个文档（doc_id="0"）滑窗，窗口序列与上游不一致
        //   （整书 4357→4691 窗），每窗边界句的上下文全变——953 处词元差异
        //   的根因。sid 取自注释行、跨文档唯一 ⇒ owner/order/账本无需按文档隔离。
        let documents = parse_jumanpp_documents_by_doc(juman_text);
        let mut windows: Vec<WordWindow> = Vec::new();
        for document in &documents {
            windows.extend(split_flat_word_document(
                document,
                self.max_token_length,
                &self.tokenizer,
                self.stride,
            )?);
        }

        let dbg = std::env::var("KWJA_TIMING").is_ok();

        // ---- 分段消化：每段 SEGMENT 个窗，段内「串行编码 → 并行解码 → 并行合并」----
        // ★为什么必须分段：每窗解码产物带 cohesion/discourse 投影矩阵
        //   （约 8–15 MB/窗），整书 4691 窗全部驻留就是几十 GB——
        //   31.6 GB 的机器实测直接内存爆掉、系统换页卡死（ 整书闸）。
        //   分段后峰值 ≈ 一段投影（64 × 15 MB ≈ 1 GB）＋ 全量小数据
        //   （chunks/sids/ranges，约 0.2 MB/窗）＋ 账本本身。
        // ★不改逐字节结果：owner/order 仍按窗序「首窗定稿」增量登记，
        //   合并仍按窗序落账；各窗合并互不依赖（见 absorb_one 注），段边界
        //   只是让投影早一点死，计算内容与全量版完全相同。
        const SEGMENT: usize = 64;
        let mut buf = Buffer::default();
        // 非跳过窗的小数据（chunks/sids/ranges），下标 = 非跳过窗序号；
        // 后续窗的合并体视图要读「owner 窗」的这三样，所以全量驻留。
        let mut keeps: Vec<WinKeep> = Vec::with_capacity(windows.len());
        // 三段耗时改用原子累计（C1 里三段在不同线程各自计时；单位微秒）
        let prep_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let dec_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let abs_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        // ★双缓冲：三级流水。编码（独占 encoder，串行 GPU）、
        //   解码（&self，rayon 并行 CPU）、合并（&self，rayon 并行 CPU）
        //   三阶段重叠：段 N 合并时，段 N+1 在解码，段 N+2 在编码。
        //   scoped thread 各占一阶段，段引用与产物走有界通道（各限 1 段，
        //   内存在界）；各段计算内容与串行版相同，逐字节结果不变。
        //   理论墙钟下限＝max(三段总耗时)，收益上限即「被掩盖的两段」。
        let pred: &WordPredictor = self; // 各线程只用共享借用（&mut self 只为兼容旧签名）
        std::thread::scope(|scope| -> Result<(), String> {
            let chunks: Vec<&[WordWindow]> = windows.chunks(SEGMENT).collect();
            let (req_tx, req_rx) = std::sync::mpsc::channel::<&[WordWindow]>();
            let (prep_tx, prep_rx) = std::sync::mpsc::sync_channel::<(
                &[WordWindow],
                Result<Vec<Option<WindowPrep>>, String>,
            )>(1);
            let (dec_tx, dec_rx) = std::sync::mpsc::sync_channel::<
                Result<Vec<Option<(WinKeep, WinBig)>>, String>,
            >(1);
            let encoder: &mut dyn Encoder = &mut *encoder;
            let prep_ms = prep_ms.clone();
            // 编码线程：独占型后端（ORT session，&mut）逐段串行；共享型后端
            //（tract 计划 Sync）把整段窗撒到 rayon 池按窗并行——与解码/合并
            // 同一个池、同一批核，不再有 ORT CPU EP 的第二线程池抢核。
            scope.spawn(move || {
                for chunk in req_rx {
                    let t = std::time::Instant::now();
                    let payload = match encoder.as_parallel() {
                        Some(par) => pred.prep_windows_parallel(par, chunk),
                        None => pred.prep_windows_serial(encoder, chunk),
                    };
                    prep_ms.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let had_err = payload.is_err();
                    if prep_tx.send((chunk, payload)).is_err() || had_err {
                        break;
                    }
                }
            });
            // 解码线程：13 头 ＋ 切分 ＋ NE ＋ 依存（窗间独立、`&self` 不写共享状态）
            let dec_ms = dec_ms.clone();
            scope.spawn(move || {
                for (chunk, preps) in prep_rx {
                    let t = std::time::Instant::now();
                    let payload = match preps {
                        Err(e) => Err(e),
                        Ok(preps) => chunk
                            .par_iter()
                            .zip(preps)
                            .map(|(window, prep)| match prep {
                                Some(p) => pred.decode_window(window, p).map(Some),
                                None => Ok(None),
                            })
                            .collect::<Result<Vec<_>, String>>(),
                    };
                    dec_ms.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let had_err = payload.is_err();
                    if dec_tx.send(payload).is_err() || had_err {
                        break;
                    }
                }
            });
            // 发全部段请求；有界通道自动背压（任一时刻至多各 1 段在途，内存有界）
            for chunk in &chunks {
                if req_tx.send(*chunk).is_err() {
                    break;
                }
            }
            drop(req_tx); // 编码线程随之收尾
            // 主线程＝合并阶段：段 N 合并与段 N+1 解码、段 N+2 编码重叠。
            // owner/order 仍按段序登记（首窗定稿），落账按窗序，逐字节结果不变。
            for _ in 0..chunks.len() {
                let outs = dec_rx.recv().map_err(|_| "解码线程意外退出".to_string())??;
                let t = std::time::Instant::now();
                let mut jobs: Vec<(usize, &WinKeep, &WinBig)> = Vec::new();
                for out in outs.iter().flatten() {
                    let (keep, big) = out;
                    let win = keeps.len();
                    register_owner(win, &keep.sids, &big.tgt_flags, &mut buf.owner, &mut buf.order);
                    keeps.push(WinKeep {
                        chunks: keep.chunks.clone(),
                        sids: keep.sids.clone(),
                        ranges: keep.ranges.clone(),
                    });
                    jobs.push((win, keep, big));
                }
                let absorbed: Vec<AbsorbedWin> = jobs
                    .par_iter()
                    .map(|(win, keep, big)| {
                        absorb_one(*win, keep, big, &keeps, &buf.owner, &pred.tables)
                    })
                    .collect();
                for a in absorbed {
                    buf.fold(a);
                }
                abs_ms.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            Ok(())
        })?;

        let t_render = std::time::Instant::now();
        let rendered = buf.render(&self.tables);
        if dbg {
            let us = |a: &std::sync::atomic::AtomicU64| {
                a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0
            };
            eprintln!(
                "TIMING prep {:.1} ms / decode {:.1} ms / absorb {:.1} ms / render {:.1} ms",
                us(&prep_ms),
                us(&dec_ms),
                us(&abs_ms),
                t_render.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(rendered)
    }

    /// 一段窗：分词 ＋ 编码器前向，**按批**跑（独占型后端：ORT session，&mut）。
    ///
    /// ★分词不碰后端，先逐窗做完；再把 batch 个窗拼成 [N, seq] 一次前向——
    ///   `encode_window` 恒定补到 max_seq_length，长度一致才能拼。与 Python 侧
    ///   同款 batch=4（环境变量 KWJA_BATCH 可覆盖）；attention mask 隔离批内
    ///   他窗，argmax 结果与逐窗前向相同（分级闸 ch1/3ch 逐字节复核）。
    ///   超长窗照 word_inference.py:118-121 跳过（不产出任何内容）。
    fn prep_windows_serial(
        &self,
        encoder: &mut dyn Encoder,
        windows: &[WordWindow],
    ) -> Result<Vec<Option<WindowPrep>>, String> {
        let n_specials = self.tables.dataset_special_tokens().len();
        let mut encodings: Vec<Option<WindowEncoding>> = Vec::with_capacity(windows.len());
        for window in windows {
            let encoding = self.tokenizer.encode_window(&window.surfs())?;
            // 超长窗：word_inference.py:118-121 warning + continue
            if encoding.num_tokens > self.max_seq_length - n_specials {
                encodings.push(None);
            } else {
                encodings.push(Some(encoding));
            }
        }
        let batch_size: usize = std::env::var("KWJA_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b| b > 0)
            .unwrap_or(4);
        let valid: Vec<usize> = (0..windows.len()).filter(|&i| encodings[i].is_some()).collect();
        let mut hiddens: Vec<ndarray::Array3<f32>> = Vec::with_capacity(valid.len());
        for group in valid.chunks(batch_size) {
            let batch = group.len();
            let mut input_ids = Vec::with_capacity(batch * self.max_seq_length);
            let mut attention_mask = Vec::with_capacity(batch * self.max_seq_length);
            let mut special_token_indices = Vec::with_capacity(batch * n_specials);
            for &wi in group {
                let enc = encodings[wi].as_ref().expect("valid 里都是已编码窗");
                input_ids.extend_from_slice(&enc.input_ids);
                attention_mask.extend_from_slice(&enc.attention_mask);
                special_token_indices
                    .extend((0..n_specials as i64).map(|i| enc.num_tokens as i64 + i));
            }
            let hidden = encoder
                .run_word(&WordEncoderInput {
                    input_ids,
                    attention_mask,
                    special_token_indices,
                    batch,
                    seq: self.max_seq_length,
                })
                .map_err(|e| e.to_string())?;
            let shape = hidden.shape();
            let (seq, hid) = (shape[1], shape[2]);
            let flat = hidden.as_slice().expect("encoder 输出应连续").to_vec();
            for k in 0..batch {
                hiddens.push(ndarray::Array3::from_shape_vec(
                    (1, seq, hid),
                    flat[k * seq * hid..(k + 1) * seq * hid].to_vec()
                )
                .map_err(|e| format!("批输出切分失败：{e}"))?);
            }
        }
        let mut hidden_iter = hiddens.into_iter();
        let mut preps = Vec::with_capacity(windows.len());
        for enc in encodings {
            match enc {
                Some(encoding) => {
                    let hidden = hidden_iter.next().expect("hiddens 与有效窗等长");
                    preps.push(Some(WindowPrep { encoding, hidden }));
                }
                None => preps.push(None)
            }
        }
        Ok(preps)
    }

    /// ★并行预备（rust-encoder-no-ort）：tract 计划 Sync，分词与编码按窗
    /// 撒到 rayon 池上并发——编码与解码/合并共用同一批核、同一个池。
    /// 每窗计算内容与串行版相同，逐字节结果不变；超长窗照旧跳过。
    fn prep_windows_parallel(
        &self,
        encoder: &dyn ParallelEncoder,
        windows: &[WordWindow],
    ) -> Result<Vec<Option<WindowPrep>>, String> {
        windows
            .par_iter()
            .map(|window| {
                let surfs = window.surfs();
                let encoding = self.tokenizer.encode_window(&surfs)?;
                let n_specials = self.tables.dataset_special_tokens().len();
                // 超长窗：word_inference.py:118-121 warning + continue
                if encoding.num_tokens > self.max_seq_length - n_specials {
                    return Ok(None);
                }
                let special_token_indices: Vec<i64> = (0..n_specials as i64)
                    .map(|i| encoding.num_tokens as i64 + i)
                    .collect();
                let hidden = encoder
                    .run_word_shared(&WordEncoderInput {
                        input_ids: encoding.input_ids.clone(),
                        attention_mask: encoding.attention_mask.clone(),
                        special_token_indices,
                        batch: 1,
                        seq: self.max_seq_length,
                    })
                    .map_err(|e| e.to_string())?;
                Ok(Some(WindowPrep { encoding, hidden }))
            })
            .collect()
    }


    /// 阶段一-B（可并行）：13 头 ＋ 切分 ＋ NE ＋ 依存候选。
    ///
    /// ★`&self`，不写任何共享状态，所以窗与窗之间可以铺满所有核。
    ///   原先这段与编码绑在同一个 `&mut self` 函数里，一次只能跑一个窗，
    ///   并行度被锁在「一个窗里的 8 个头」，8 个任务用 12 个线程，占用率上不去。
    fn decode_window(
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
        // ★reading 吃 hidden_rows、其余头吃 pooled，互不依赖；
        //   reading 有 12907 个输出类别，单跑就 85 ms，是组里最大的一块。
        //   组内只算 logits/概率，argmax 与后续归并留在组外串行做——
        //   那些依赖前一步结果，并行没有意义。
        let h = &self.heads;
        // ★九个头的耗时不均匀（reading 85 ms、bp_feature 109 ms、其余 6–20 ms），
        //   二叉 join 会把大块和小块切到一起，rayon 只能按切分树并行，
        //   实测 CPU 只用四成。改为扁平任务表交给 par_iter，由线程池自己调度。
        //   ★必须是 IndexedParallelIterator（用 collect 到 Vec），顺序才有保证。
        const T_READING: usize = 0;
        const T_POS: usize = 1;
        const T_SUBPOS: usize = 2;
        const T_CONJTYPE: usize = 3;
        const T_CONJFORM: usize = 4;
        const T_WF: usize = 5;
        const T_NE: usize = 6;
        const T_BF: usize = 7;
        // ★计算量方向：这些头下游只读词素行——
        //   pos/subpos/conjtype/conjform 下游显式取 [..n_morphemes]，
        //   wf 的 chunk_plain、bf 的 add_bp_features 也只按词素/主辞下标取行，
        //   reading 只被 rmap 里真词素的 token 位读（填充与 special 位从不读）。
        //   forward_rows 是逐行 matvec、行与行独立 ⇒ 只算这些行，结果逐字节相同。
        //   NE 例外：它走全序列 Viterbi（掩码路径跨整窗），保持原样。
        let token_rows = &hidden_rows[..encoding.num_tokens];
        let morph_rows = &pooled[..n_morphemes];
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
        let reading_logits = &outs[T_READING];
        let wf_prob = &outs[T_WF];
        let ne_logits = &outs[T_NE];
        let bf_prob = &outs[T_BF];
        let attrs = get_morpheme_attribute_predictions(
            &outs[T_POS][..n_morphemes],
            &outs[T_SUBPOS][..n_morphemes],
            &outs[T_CONJTYPE][..n_morphemes],
            &outs[T_CONJFORM][..n_morphemes],
        );

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


        // ---- 组装词素（词典/jinf/同形词；readings 即模型读音，同时用于
        //      意味素性查找与 KNP 读方字段）----
        if dbg4 {
            eprintln!("      encoder+heads {:.1} ms", t_enc.elapsed().as_secs_f64() * 1000.0);
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
            eprintln!("      build_morphemes {:.1} ms", t_build.elapsed().as_secs_f64() * 1000.0);
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
        let sids: Vec<String> = window.sentences.iter().map(|s| strip_subdoc(&s.sid)).collect();

        // ---- 切分 ----
        let t_chunk = std::time::Instant::now();
        let mut chunks = chunk_plain(&mut flat, &ranges, &wf_prob, &self.tables.word_features);
        if dbg4 {
            eprintln!("      chunk_plain {:.1} ms", t_chunk.elapsed().as_secs_f64() * 1000.0);
        }
        let t_ne = std::time::Instant::now();

        // ---- NE（只作用目标句）----
        //
        // ★掩码是**整窗全部目标句**，不是当前这一句。
        //   Python 侧的 ne_mask 来自 target_mask，按 analysis_target_morpheme_indices
        //   一次性标满（word_inference.py:175-177、:233），Viterbi 跑**一次**。
        //   原先这里逐句重建掩码、逐句跑 Viterbi，于是每句都被当成「窗口里唯一的目标段」，
        //   解码路径就变了：一个窗只有一个目标句时两者恰好等价（第 1 章没暴露），
        //   一窗多个目标句时就多报实体——3 章闸实测「昨日」多出 <NE:DATE>。
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

        if dbg4 {
            eprintln!("      NE+bp_features {:.1} ms", t_ne.elapsed().as_secs_f64() * 1000.0);
        }
        let t_dep = std::time::Instant::now();
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
            // ★行与行独立，并行算（dependency 单窗 70–800 ms，是这里最大的一块）。
            //   par_iter 保序，dep_preds 的下标与串行一致。
            let rows: Vec<(usize, Vec<DepCandidate>)> = head_gis
                .par_iter()
                .map(|&hg| {
                    let row = dep_proj.row(hg);
                    let mut masked: Vec<f32> = Vec::with_capacity(self.max_seq_length);
                    let row_mask =
                        &mask[hg * self.max_seq_length..(hg + 1) * self.max_seq_length];
                    for (j, &ok) in row_mask.iter().enumerate() {
                        masked.push(if ok { row[j][0] } else { kwja_heads::mask_tables::MASKED });
                    }
                    let mut cands = Vec::new();
                    for parent_gi in topk_indices(&masked, self.tables.dependency_topk) {
                        let dtype = self
                            .heads
                            .dependency_type
                            .predict_pair(&pooled[hg], &pooled[parent_gi]);
                        cands.push(DepCandidate { parent_gi, dep_type_idx: dtype });
                    }
                    (hg, cands)
                })
                .collect();
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
            eprintln!("      dependency {:.1} ms", t_dep.elapsed().as_secs_f64() * 1000.0);
        }
        let t_proj = std::time::Instant::now();
        // ---- cohesion / discourse 的投影（解码在缓冲合并体上做）----
        let cohesion_proj = self.heads.cohesion.project(&pooled);
        let discourse_proj = self.heads.discourse.project(&pooled);

        // 词素 → 句序（cohesion 同句判据）
        let mut sent_of_morpheme = vec![0usize; self.max_seq_length];
        for (si, (lo, hi)) in ranges.iter().enumerate() {
            sent_of_morpheme[*lo..*hi].fill(si);
        }

        if dbg4 {
            eprintln!("      projections {:.1} ms", t_proj.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((
            WinKeep { chunks, sids, ranges },
            WinBig {
                flat,
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
            },
        ))
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

/// 阶段一-A 的产物：分词结果 ＋ 编码器输出。
struct WindowPrep {
    encoding: WindowEncoding,
    hidden: ndarray::Array3<f32>,
}

/// 解码产物·跨窗保留的小数据：后续窗的合并体视图要读 owner 窗的这三样。
/// 全量驻留也只有约 0.2 MB/窗；大投影不在此列（随段释放，见 predict_text）。
struct WinKeep {
    chunks: Vec<Chunk>,
    sids: Vec<String>,
    ranges: Vec<(usize, usize)>,
}

/// 解码产物·只在本窗合并期用的大件（投影矩阵约 8–15 MB/窗），段末即释放。
struct WinBig {
    flat: Vec<FlatMorpheme>,
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

/// 留档的窗口数据（渲染用；合并期的 ranges/tgt_flags 不进留档）。
struct StoredWin {
    flat: Vec<FlatMorpheme>,
    chunks: Vec<Chunk>,
    sids: Vec<String>,
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

/// 一次窗合并的产物：渲染要用的本窗留档 ＋ 三类行。
/// 由 [`absorb_one`] 按窗并行产出，[`Buffer::fold`] 按窗序落账。
struct AbsorbedWin {
    win: StoredWin,
    rel_rows: HashMap<(String, usize), Vec<RelRow>>,
    dis_rows: HashMap<(String, usize), String>,
    dep_rows: HashMap<String, (Vec<i32>, Vec<usize>)>,
}

/// owner/order 登记（原串行 absorb 第 1/7 步）：sid 的 owner ＝ 它首次作为
/// 目标句出现的窗（**非跳过窗**里的序号，与渲染时的 `wins` 下标一致）。
/// 按窗序逐窗调用即等价于原来的一次性预计算；目标句集合按窗不重叠，
/// 首窗定稿语义不变。
fn register_owner(
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

/// 单窗合并（原 `Buffer::absorb` 的第 2–6 步）。
///
/// ★只读三样：本窗解码产物（`keep`＋`big`）、已登记的 `owner`、之前各窗的
///   小数据 `live`（合并体视图里缓冲句的 chunk 来源）。不碰任何累积账本 ⇒
///   窗与窗可并行。产出的行按窗序 [`Buffer::fold`] 落账；目标键按窗不重叠，
///   落账顺序不影响结果。
fn absorb_one(
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

    // 4) cohesion（目标 = 本窗目标句的基本句；候选 = 合并体全部基本句＋special）
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
    // 本窗目标键此前不存在于账本（键按窗唯一），原先「先清再加」在落账侧做
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
        let rows = add_cohesion(
            &doc,
            &|gi: usize, cols: &[usize]| cohesion_proj.row_gathered(gi, cols),
            &tables.cohesion_tasks,
            &|t: &str| tables.cohesion_task2name(t),
            &|t: &str| tables.cohesion_task2rels(t),
            &|a: usize, c: usize| sent_of_morpheme[a] == sent_of_morpheme[c],
            &big.cohesion_special_indices,
            &tables.dataset_special_tokens().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
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
    /// 按窗序落账：把 [`absorb_one`] 的产物并进累积账本。
    /// 目标键按窗不重叠 ⇒ rel/dis 的 insert 等价于原先的「先清再加」；
    /// dep 保留 `or_insert` 的首窗定稿语义。wins 必须按窗序 push
    /// （渲染按 owner 下标取窗）。
    fn fold(&mut self, a: AbsorbedWin) {
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
