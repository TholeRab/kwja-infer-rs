//! word 阶段推理编排：Juman++ 文本经 13 头解码得到 KNP 文本。
//!
//! 流程照抄参考实现的扁平链：
//! 1. 解析与分窗（split_flat_word_document）
//! 2. 每窗：编码（special 追加加 pad）、ONNX encoder、FIRST 池化、13 头
//! 3. 解码：读音（token 级 argmax 后按词归并）、四路属性（掩码 argmax）、
//!    NE（CRF Viterbi）、词素/基本句素性（多标签 sigmoid 阈值）、
//!    依存（topk、类型 argmax、成环回退）、cohesion（掩码 gather argmax）、
//!    談話関係（softmax 阈值）
//! 4. 组装：切分（chunk_plain）、三小步（只作用目标句）、合并体视图
//!    （缓冲句用首次那窗的结构，坐标平移对齐）、cohesion/discourse
//! 5. 缓冲：首窗定稿（first-window-wins），按插入序渲染 KNP

use rayon::prelude::*;
use std::path::Path;

use kwja_encoder::Encoder;
use kwja_heads::forward::WordHeads;
use kwja_heads::sequence_labeling::CrfParameters;
use kwja_tokenizer::word_side::WordTokenizer;

use crate::jinf::Jinf;
use crate::jumandic::JumanDic;
use crate::word_decode::{WinBig, WinKeep};
use crate::word_encode::WindowPrep;
use crate::word_merge::{AbsorbedWin, Buffer, absorb_one, register_owner};
use crate::word_stage::{WordWindow, parse_jumanpp_documents_by_doc, split_flat_word_document};
use crate::word_tables::WordTables;

/// word 阶段推理器。编码器按调用传入（与 char_stage 同风格）。
pub struct WordPredictor {
    pub(crate) heads: WordHeads,
    pub(crate) tables: WordTables,
    pub(crate) jumandic: JumanDic,
    pub(crate) jinf: Jinf,
    pub(crate) tokenizer: WordTokenizer,
    pub(crate) max_seq_length: usize,
    pub(crate) max_token_length: usize,
    pub(crate) stride: i64,
    pub(crate) crf: CrfParameters,
    /// 头前向的 GPU 后端（ORT 融合图或 cuBLAS 直调）；权重缺失或 CPU 设备时为 None。
    pub(crate) gpu_heads: Option<kwja_heads::gpu_backend::GpuHeadsBackend>,
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
        // tokenizer：8 个 additional specials（含 " "）加数据集的 7 个
        let additional: Vec<&str> = tables.special_tokens.iter().map(|s| s.as_str()).collect();
        let dataset_specials = tables.dataset_special_tokens();
        let tokenizer = WordTokenizer::load(
            tokenizer_json(assets_dir)?,
            &additional,
            &dataset_specials,
            max_seq_length,
        )?;
        // max_token_length = max_seq_length - len(extra_special_tokens) - 2
        //（extra 即 8 个 additional specials）
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
            gpu_heads: None,
        })
    }

    /// 把头前向切到 GPU 后端（ORT 融合图或 cuBLAS 直调）。
    pub fn set_gpu_heads(&mut self, gpu_heads: Option<kwja_heads::gpu_backend::GpuHeadsBackend>) {
        self.gpu_heads = gpu_heads;
    }

    /// 头前向现在跑在哪：`None` = CPU，`Some("cublas"|"ort-cuda")` = GPU。
    ///
    /// ★**为什么要有这个 getter**： 我写的公版语料测试漏了
    ///   `set_gpu_heads`，实际跑成「编码器在 GPU、13 个头在 CPU」的混跑状态——
    ///   **它不报错、产出也对，只是慢**，靠肉眼看日志才发现。
    ///   有了这个查询接口，测试可以断言「要求全 GPU 时头真的在 GPU」，
    ///   把这类「静默降级」变成当场失败。
    #[must_use]
    pub fn head_device(&self) -> Option<&'static str> {
        match self.gpu_heads {
            None => None,
            Some(kwja_heads::gpu_backend::GpuHeadsBackend::Cublas(_)) => Some("cublas"),
            Some(kwja_heads::gpu_backend::GpuHeadsBackend::Ort(_)) => Some("ort-cuda"),
        }
    }

    /// 一份 char 产出（Juman++ 文本）转成 KNP 文本（对应一次 predict 批）。
    pub fn predict_text(
        &mut self,
        encoder: &mut dyn Encoder,
        juman_text: &str,
    ) -> Result<String, String> {
        // 必须按文档切分后再逐文档滑窗（见 parse_jumanpp_documents_by_doc）。
        // 若把整份输入当一个文档滑窗，窗口序列与参考实现不一致，
        // 每窗边界句的上下文全变。sid 取自注释行、跨文档唯一，
        // owner/order/缓冲区无需按文档隔离。
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

        // ---- 分段消化：每段 SEGMENT 个窗，段内串行编码、并行解码、并行合并 ----
        // 为什么必须分段：每窗解码产物带 cohesion/discourse 投影矩阵
        //   （约 8-15 MB/窗），全量驻留会到几十 GB，内存装不下。
        //   分段后峰值约为一段投影（64 窗、每窗约 15 MB，约 1 GB）加全量小数据
        //   （chunks/sids/ranges，约 0.2 MB/窗）加缓冲区本身。
        // 不改逐字节结果：owner/order 仍按窗序「首窗定稿」增量登记，
        //   合并仍按窗序写入；各窗合并互不依赖（见 absorb_one 注），段边界
        //   只是让投影早一点死，计算内容与全量版完全相同。
        const SEGMENT: usize = 64;
        let mut buf = Buffer::default();
        // 流式渲出的文本，按章拼接。见 stream_release 的说明。
        let mut out_text = String::new();
        // 非跳过窗的小数据（chunks/sids/ranges），下标 = 非跳过窗序号；
        // 其他窗的合并体视图要读「owner 窗」的这三样，所以全量驻留。
        // ★Vec<Option<..>>：按章渲完后把该章的槽位置 None 释放内存，
        //   但**下标不压缩**——absorb_one 的 live[ow] 用绝对窗口序号索引。
        let mut keeps: Vec<Option<WinKeep>> = Vec::with_capacity(windows.len());
        // 三段耗时改用原子累计（三段在不同线程各自计时；单位微秒）
        let prep_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let dec_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let abs_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        // 三级流水：编码（独占 encoder，串行 GPU）、
        //   解码（&self，rayon 并行 CPU）、合并（&self，rayon 并行 CPU）
        //   三段重叠：段 N 合并时，段 N+1 在解码，段 N+2 在编码。
        //   scoped thread 各占一段，段引用与产物走有界通道（各限 1 段，
        //   内存在界）；各段计算内容与串行版相同，逐字节结果不变。
        let pred: &WordPredictor = self; // 各线程只用共享借用（&mut self 只为兼容旧签名）
        std::thread::scope(|scope| -> Result<(), String> {
            let chunks: Vec<&[WordWindow]> = windows.chunks(SEGMENT).collect();
            let (req_tx, req_rx) = std::sync::mpsc::channel::<&[WordWindow]>();
            let (prep_tx, prep_rx) = std::sync::mpsc::sync_channel::<(
                &[WordWindow],
                Result<Vec<Option<WindowPrep>>, String>,
            )>(1);
            let (dec_tx, dec_rx) =
                std::sync::mpsc::sync_channel::<Result<Vec<Option<(WinKeep, WinBig)>>, String>>(1);
            let encoder: &mut dyn Encoder = &mut *encoder;
            let prep_ms = prep_ms.clone();
            // 编码线程：独占型后端（ORT session，&mut）逐段串行；共享型后端
            //（tract 后端，实现 Sync）把整段窗撒到 rayon 池按窗并行——与解码/合并
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
            // 解码线程：13 头 + 切分 + NE + 依存（窗间独立、&self 不写共享状态）
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
            // 主线程做合并：段 N 合并与段 N+1 解码、段 N+2 编码重叠。
            // owner/order 仍按段序登记（首窗定稿），结果按窗序写入，逐字节结果不变。
            for seg_idx in 0..chunks.len() {
                let outs = dec_rx
                    .recv()
                    .map_err(|_| "解码线程意外退出".to_string())??;
                let t = std::time::Instant::now();

                // ---- 补记被跳过的句子（超长窗） ----
                // outs 与 chunks[seg_idx] 同序：outs[i] 是 None 表示该窗词素超限被跳过。
                // 上游 kwja 在这里只是一句 `continue`，丢掉整窗且不吭声（PR #264 确认
                // 「超长不分析」是其本意）；我们不改变这个分析口径，但**必须让人看得见丢了什么**，
                // 所以把跳过窗里的句子整句留档，渲染成一条「未定義語」词素行。
                {
                    let seg = chunks[seg_idx];
                    let mut skipped_now: Vec<crate::word_merge::SkippedSentence> = Vec::new();
                    for (i, out) in outs.iter().enumerate() {
                        if out.is_none() && i < seg.len() {
                            for s in &seg[i].sentences {
                                // 表层直接拼接。**段落边界在这里拿不到，也不需要**：
                                // 一个 S-ID 可能跨段（实测第 9 章「時下秋冷」那句跨 4 段），
                                // 但这是全链路的既有行为——char 阶段输出 Juman++ 时就不带换行，
                                // 正常分析的句子同样还原不出段落。本块只要与正常句同口径即可。
                                skipped_now.push(crate::word_merge::SkippedSentence {
                                    sid: s.sid.clone(),
                                    text: s.surfs.concat(),
                                });
                            }
                        }
                    }
                    if !skipped_now.is_empty() {
                        eprintln!(
                            "警告：word 模块跳过了 {} 句（词素超过窗口上限，未做形态论分析；KNP 中以未定義語行保留原文）",
                            skipped_now.len()
                        );
                        buf.push_skipped(skipped_now);
                    }
                }
                let mut jobs: Vec<(usize, &WinKeep, &WinBig)> = Vec::new();
                for out in outs.iter().flatten() {
                    let (keep, big) = out;
                    let win = keeps.len();
                    register_owner(
                        win,
                        &keep.sids,
                        &big.tgt_flags,
                        &mut buf.owner,
                        &mut buf.order,
                    );
                    keeps.push(Some(WinKeep {
                        chunks: keep.chunks.clone(),
                        sids: keep.sids.clone(),
                        ranges: keep.ranges.clone(),
                    }));
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

                // ---- 流式渲出已完整的章 ----
                // sid 形如 c-07-49，章前缀是最后一个 '-' 之前的部分。
                // 窗口按章连续（实测章切换恰好比章数少一），所以「order 里同一个章前缀
                // 连续出现」成立：一旦 order 末尾那个章前缀不再增长，前面那些章就齐了。
                // 判断「不再增长」的成本很低——只是比较字符串，不扫全表。
                stream_release(&mut buf, &mut keeps, &pred.tables, &mut out_text);
            }
            Ok(())
        })?;

        let t_render = std::time::Instant::now();
        // 收尾：把剩下的（最后一章及任何未渲的）全部渲出
        out_text.push_str(&buf.render_and_release(buf.order_len(), &self.tables));
        let rendered = std::mem::take(&mut out_text);
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
}

/// 流式渲出「已完整的章」并释放其窗口。
///
/// ★为什么按章：sid 形如 `c-07-49`，章前缀 = 去掉末尾 `-N`。
///   实测 sid 全书不重复、窗口严格按章连续（章切换恰好比章数少一），
///   所以 `Buffer::order` 里同一章前缀是连续的一段，前面各段一定是完整的章。
///
/// ★为什么能释放：`render` 对每个 sid 只读 `wins[owner[sid]]` 与三个行表，
///   键都是 sid 或 (sid, idx)，sid 章内唯一 ⇒ 不存在跨章引用。
///
/// `keeps` 这里**不裁剪**：它按绝对窗口序号被 `absorb_one` 的 `live[ow]` 索引，
///   裁了会错位。它约 0.2 MB/窗，比 `wins` 小一个数量级，先不动。
fn stream_release(
    buf: &mut Buffer,
    keeps: &mut [Option<WinKeep>],
    tables: &crate::word_tables::WordTables,
    out: &mut String,
) {
    let n = buf.completed_prefix_len();
    if n == 0 {
        return;
    }
    // 先渲：渲染只读 order[..n] 与 wins[owner[sid]]
    out.push_str(&buf.render_and_release(n, tables));
    // 再释放该批窗口：owner 已随 render_and_release 清掉，
    // 这里把「已不再被任何存活 sid 引用」的 keeps 槽位置 None。
    // 判据与 render_and_release 内部一致：owner 里已无该窗 ⇒ 可丢。
    let live: std::collections::HashSet<usize> = buf.owner.values().copied().collect();
    for (w, slot) in keeps.iter_mut().enumerate() {
        if slot.is_some() && !live.contains(&w) {
            *slot = None;
        }
    }
}

/// tokenizer.json 定位（snapshot 目录唯一）。
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
