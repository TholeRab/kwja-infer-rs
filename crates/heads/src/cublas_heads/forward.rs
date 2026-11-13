//! 一窗融合前向与 dep_type argmax。

use crate::gpu_heads::FusedHeadsOutput;
use crate::sequence_labeling::argmax;

use super::ffi::Ptr;
use super::ops::unsafe_fn_copy;
use super::{Inner, OutLayout};

fn out_layout(t: usize, m: usize) -> OutLayout {
    let mut o = 0usize;
    let rargmax = { let v = o; o += t; v };
    let pos = { let v = o; o += m * 14; v };
    let subpos = { let v = o; o += m * 35; v };
    let conjtype = { let v = o; o += m * 33; v };
    let conjform = { let v = o; o += m * 81; v };
    let wf = { let v = o; o += m * 5; v };
    let bpf = { let v = o; o += m * 64; v };
    let ne = { let v = o; o += m * 17; v };
    let dep_hs = { let v = o; o += m * 768; v };
    let dep_ht = { let v = o; o += m * 768; v };
    let dis_hs = { let v = o; o += m * 768; v };
    let dis_ht = { let v = o; o += m * 768; v };
    let coh_s = { let v = o; o += m * 6912; v };
    let coh_t = { let v = o; o += m * 6912; v };
    OutLayout { rargmax, pos, subpos, conjtype, conjform, wf, bpf, ne, dep_hs, dep_ht, dis_hs, dis_ht, coh_s, coh_t, total: o }
}

impl Inner {
pub(super)    fn forward_fused(&mut self, token_rows: &[Vec<f32>], pooled_rows: &[Vec<f32>]) -> Result<FusedHeadsOutput, String> {
        unsafe {
            // ORT 等其它 CUDA 使用方可能切换了当前上下文；cuBLAS 句柄绑定在本上下文，
            // 每次前向先归位，防止跨上下文调用产生垃圾输出。
            if (self.cu_ctx_set)(self.ctx) != 0 {
                return Err("cuCtxSetCurrent（前向归位）失败".into());
            }
            let t = token_rows.len();
            let m = pooled_rows.len();
            if t == 0 || m == 0 {
                return Err("空窗口".into());
            }
            let lay = out_layout(t, m);

            // 进卡：hidden/pooled 摊平进 pinned，一次 H2D
            let in_floats = 768 * (t + m);
            let pin_in = self.pin("in", in_floats * 4)?;
            let pin_in_f = pin_in as *mut f32;
            Self::flatten_into(token_rows, 768, pin_in_f, "token_rows")?;
            Self::flatten_into(pooled_rows, 768, pin_in_f.add(t * 768), "pooled_rows")?;
            let hidden = self.buf("hidden", t * 768 * 4)?;
            let pooled = self.buf("pooled", m * 768 * 4)?;
            self.to_device(pin_in, hidden, t * 768 * 4)?;
            self.to_device((pin_in as *mut u8).add(t * 768 * 4) as Ptr, pooled, m * 768 * 4)?;

            // 输出缓冲：一窗一次回读，布局见 out_layout
            let outs = self.buf("outs", lay.total * 4)?;
            let outs_f = outs as *mut f32;

            // reading：Linear(768,768)+GELU+Linear(768,12907)，逐行 argmax
            let r1 = self.buf("r1", t * 768 * 4)?;
            let (rw1, rb1) = (self.w("reading_w1")?, self.w("reading_b1")?);
            self.gemm_nn(hidden, rw1.ptr, r1, t, 768, 768)?;
            self.bias_gelu(r1, rb1.ptr, t * 768, 768)?;
            let rlogits = self.buf("rlogits", t * 12907 * 4)?;
            let (rw2, rb2) = (self.w("reading_w2")?, self.w("reading_b2")?);
            self.gemm_nn(r1, rw2.ptr, rlogits, t, 768, 12907)?;
            self.bias_add(rlogits, rb2.ptr, t * 12907, 12907)?;
            self.row_argmax(rlogits, 12907, t, outs_f.add(lay.rargmax) as Ptr)?;

            // 四个小 tagger：pooled 经 Linear(768,768)+GELU+Linear(768,C)
            let tag1 = self.buf("tag1", m * 768 * 4)?;
            let mut run_tagger = |g: &mut Self, w1n: &str, b1n: &str, w2n: &str, b2n: &str, c: usize, dst: Ptr| -> Result<(), String> {
                let (w1, b1) = (g.w(w1n)?, g.w(b1n)?);
                g.gemm_nn(pooled, w1.ptr, tag1, m, 768, 768)?;
                g.bias_gelu(tag1, b1.ptr, m * 768, 768)?;
                let (w2, b2) = (g.w(w2n)?, g.w(b2n)?);
                g.gemm_nn(tag1, w2.ptr, dst, m, 768, c)?;
                g.bias_add(dst, b2.ptr, m * c, c)?;
                Ok(())
            };
            run_tagger(self, "pos_w1", "pos_b1", "pos_w2", "pos_b2", 14, outs_f.add(lay.pos) as Ptr)?;
            run_tagger(self, "subpos_w1", "subpos_b1", "subpos_w2", "subpos_b2", 35, outs_f.add(lay.subpos) as Ptr)?;
            run_tagger(self, "conjtype_w1", "conjtype_b1", "conjtype_w2", "conjtype_b2", 33, outs_f.add(lay.conjtype) as Ptr)?;
            run_tagger(self, "conjform_w1", "conjform_b1", "conjform_w2", "conjform_b2", 81, outs_f.add(lay.conjform) as Ptr)?;

            // LoRA 双头（wf 5 类 / bpf 64 类）：base 线性加 delta，广播加后过 GELU，再过 classifier 与 sigmoid
            let mut run_lora = |g: &mut Self, w1n: &str, b1n: &str, delta: &str, clf: &str, b2n: &str, l: usize, slot_base: &'static str, slot_g: &'static str, dst: Ptr| -> Result<(), String> {
                let base = g.buf(slot_base, m * 768 * 4)?;
                let (w1, b1) = (g.w(w1n)?, g.w(b1n)?);
                g.gemm_nn(pooled, w1.ptr, base, m, 768, 768)?;
                g.bias_add(base, b1.ptr, m * 768, 768)?;
                let lora = g.buf("lora", m * 768 * l * 4)?;
                let d = g.w(delta)?;
                g.gemm_nn(pooled, d.ptr, lora, m, 768, 768 * l)?;
                let gs = g.buf(slot_g, m * 768 * l * 4)?;
                g.bcast_add_gelu(base, lora, gs, m * 768 * l, l)?;
                let (cw, cb) = (g.w(clf)?, g.w(b2n)?);
                g.classify_sigmoid(gs, cw.ptr, cb.ptr, dst, m, 768, l)?;
                Ok(())
            };
            run_lora(self, "wf_w1", "wf_b1", "wf_delta", "wf_clf", "wf_b2", 5, "wf_base", "wf_g", outs_f.add(lay.wf) as Ptr)?;
            run_lora(self, "bpf_w1", "bpf_b1", "bpf_delta", "bpf_clf", "bpf_b2", 64, "bpf_base", "bpf_g", outs_f.add(lay.bpf) as Ptr)?;

            // ne：pooled 经 Linear(768,768)+GELU+Linear(768,17)
            let ne1 = self.buf("ne1", m * 768 * 4)?;
            let (nw1, nb1) = (self.w("ne_w1")?, self.w("ne_b1")?);
            self.gemm_nn(pooled, nw1.ptr, ne1, m, 768, 768)?;
            self.bias_gelu(ne1, nb1.ptr, m * 768, 768)?;
            let (nw2, nb2) = (self.w("ne_w2")?, self.w("ne_b2")?);
            self.gemm_nn(ne1, nw2.ptr, outs_f.add(lay.ne) as Ptr, m, 768, 17)?;
            self.bias_add(outs_f.add(lay.ne) as Ptr, nb2.ptr, m * 17, 17)?;

            // dep/discourse 投影：单层线性，无激活
            let mut run_proj = |g: &mut Self, wn: &str, bn: &str, dst: Ptr| -> Result<(), String> {
                let (w, b) = (g.w(wn)?, g.w(bn)?);
                g.gemm_nn(pooled, w.ptr, dst, m, 768, 768)?;
                g.bias_add(dst, b.ptr, m * 768, 768)?;
                Ok(())
            };
            run_proj(self, "dep_ws", "dep_bs", outs_f.add(lay.dep_hs) as Ptr)?;
            run_proj(self, "dep_wt", "dep_bt", outs_f.add(lay.dep_ht) as Ptr)?;
            run_proj(self, "dis_ws", "dis_bs", outs_f.add(lay.dis_hs) as Ptr)?;
            run_proj(self, "dis_wt", "dis_bt", outs_f.add(lay.dis_ht) as Ptr)?;

            // cohesion：单层线性 + per-rel LoRA（广播相加），输出 (M,9,768) 平坦
            let mut run_coh = |g: &mut Self, wn: &str, bn: &str, dn: &str, slot_base: &'static str, slot_delta: &'static str, dst: Ptr| -> Result<(), String> {
                let base = g.buf(slot_base, m * 768 * 4)?;
                let (w, b) = (g.w(wn)?, g.w(bn)?);
                g.gemm_nn(pooled, w.ptr, base, m, 768, 768)?;
                g.bias_add(base, b.ptr, m * 768, 768)?;
                let delta = g.buf(slot_delta, m * 6912 * 4)?;
                let dw = g.w(dn)?;
                g.gemm_nn(pooled, dw.ptr, delta, m, 768, 6912)?;
                g.coh_bcast_add(base, delta, dst, m * 6912, 9, 768)?;
                Ok(())
            };
            run_coh(self, "coh_ws", "coh_bs", "coh_delta_s", "coh_base_s", "coh_delta_s_buf", outs_f.add(lay.coh_s) as Ptr)?;
            run_coh(self, "coh_wt", "coh_bt", "coh_delta_t", "coh_base_t", "coh_delta_t_buf", outs_f.add(lay.coh_t) as Ptr)?;

            if (self.sync)() != 0 {
                return Err("cudaDeviceSynchronize 失败".into());
            }

            // 一次 D2H 回读全部输出，CPU 侧装配
            let pin_out = self.pin("out", lay.total * 4)?;
            self.to_host(outs, pin_out, lay.total * 4)?;
            let host = std::slice::from_raw_parts(pin_out as *const f32, lay.total);

            let reading_argmax: Vec<usize> = {
                let bytes = std::slice::from_raw_parts((pin_out as *mut u8).add(lay.rargmax * 4) as *const u8, t * 4);
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                    .collect()
            };
            let mut pos_logits = Vec::with_capacity(m);
            let mut subpos_logits = Vec::with_capacity(m);
            let mut conjtype_logits = Vec::with_capacity(m);
            let mut conjform_logits = Vec::with_capacity(m);
            let mut wf_prob = Vec::with_capacity(m);
            let mut bpf_prob = Vec::with_capacity(m);
            let mut ne_logits = Vec::with_capacity(m);
            Self::rows_from(&mut pos_logits, &host[lay.pos..lay.pos + m * 14], m, 14);
            Self::rows_from(&mut subpos_logits, &host[lay.subpos..lay.subpos + m * 35], m, 35);
            Self::rows_from(&mut conjtype_logits, &host[lay.conjtype..lay.conjtype + m * 33], m, 33);
            Self::rows_from(&mut conjform_logits, &host[lay.conjform..lay.conjform + m * 81], m, 81);
            Self::rows_from(&mut wf_prob, &host[lay.wf..lay.wf + m * 5], m, 5);
            Self::rows_from(&mut bpf_prob, &host[lay.bpf..lay.bpf + m * 64], m, 64);
            Self::rows_from(&mut ne_logits, &host[lay.ne..lay.ne + m * 17], m, 17);
            let mut dep_hs = Vec::with_capacity(m);
            let mut dep_ht = Vec::with_capacity(m);
            let mut dis_hs = Vec::with_capacity(m);
            let mut dis_ht = Vec::with_capacity(m);
            Self::rows_from(&mut dep_hs, &host[lay.dep_hs..lay.dep_hs + m * 768], m, 768);
            Self::rows_from(&mut dep_ht, &host[lay.dep_ht..lay.dep_ht + m * 768], m, 768);
            Self::rows_from(&mut dis_hs, &host[lay.dis_hs..lay.dis_hs + m * 768], m, 768);
            Self::rows_from(&mut dis_ht, &host[lay.dis_ht..lay.dis_ht + m * 768], m, 768);

            Ok(FusedHeadsOutput {
                reading_argmax,
                pos_logits,
                subpos_logits,
                conjtype_logits,
                conjform_logits,
                wf_prob,
                bpf_prob,
                ne_logits,
                dep_hs,
                dep_ht,
                coh_source: host[lay.coh_s..lay.coh_s + m * 6912].to_vec(),
                coh_target: host[lay.coh_t..lay.coh_t + m * 6912].to_vec(),
                dis_hs,
                dis_ht,
            })
        }
    }

pub(super)    fn dep_type_argmax(&mut self, pairs: &[(usize, usize)], pooled: &[Vec<f32>]) -> Result<Vec<usize>, String> {
        unsafe {
            if (self.cu_ctx_set)(self.ctx) != 0 {
                return Err("cuCtxSetCurrent（dep_type 归位）失败".into());
            }
            let p = pairs.len();
            if p == 0 {
                return Ok(Vec::new());
            }
            const PAIR_DIM: usize = 1536;
            let pin = self.pin("dt_pairs", p * PAIR_DIM * 4)?;
            let pin_f = pin as *mut f32;
            for (i, (hg, parent)) in pairs.iter().enumerate() {
                let base = pin_f.add(i * PAIR_DIM);
                unsafe_fn_copy(base, &pooled[*hg]);
                unsafe_fn_copy(base.add(768), &pooled[*parent]);
            }
            let pairs_dev = self.buf("dt_pairs_dev", p * PAIR_DIM * 4)?;
            self.to_device(pin, pairs_dev, p * PAIR_DIM * 4)?;
            let mid = self.buf("dt_mid", p * PAIR_DIM * 4)?;
            let (w0, b0) = (self.w("dt_w0")?, self.w("dt_b0")?);
            self.gemm_tn(pairs_dev, w0.ptr, mid, p, PAIR_DIM, PAIR_DIM)?;
            self.bias_add(mid, b0.ptr, p * PAIR_DIM, PAIR_DIM)?;
            self.gelu(mid, p * PAIR_DIM)?;
            let out = self.buf("dt_out", p * 4 * 4)?;
            let (w3, b3) = (self.w("dt_w3")?, self.w("dt_b3")?);
            self.gemm_tn(mid, w3.ptr, out, p, PAIR_DIM, 4)?;
            self.bias_add(out, b3.ptr, p * 4, 4)?;
            if (self.sync)() != 0 {
                return Err("cudaDeviceSynchronize 失败".into());
            }
            let pin_out = self.pin("dt_out_host", p * 4 * 4)?;
            self.to_host(out, pin_out, p * 4 * 4)?;
            let host = std::slice::from_raw_parts(pin_out as *const f32, p * 4);
            Ok((0..p).map(|i| argmax(&host[i * 4..i * 4 + 4])).collect())
        }
    }
}

