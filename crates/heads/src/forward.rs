//! word 阶段 13 个头的**前向**（阶段 7 落地）：权重从 checkpoint 导出的
//! safetensors 读入（`assets/models/word_heads.safetensors`，由
//! `导出工具/导出Rust侧阶段7权重与基准.py` 生成）。
//!
//! 算术与 torch 逐式对应：`Linear = x·W^T + b`、
//! `GELU = 0.5·x·(1+erf(x/√2))`（torch `nn.GELU` 默认 `approximate='none'`）、
//! 多标签头最后 `sigmoid`。pairwise 头（dependency/cohesion/discourse）按
//! 「谓词行」惰性计算：每个 logit 元素的算术只依赖 (source i, target j) 这一对，
//! 只算解码要读的行，结果与整张算完逐元素相同（阶段 9 的剪枝再进一步缩范围）。
//!
//! safetensors 读入是手写的极简解析：8 字节小端头长 ＋ JSON 头 ＋ 数据区，
//! 只认 `F32`；不值得为这一个格式引依赖。权重整文件读入后按名拷出。

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::mask_tables::{CONJFORM_TAGS, CONJTYPE_TAGS, POS_TAGS, SUBPOS_TAGS};

// ---------- safetensors 极简读取 ----------

struct WeightFile {
    bytes: Vec<u8>,
    entries: HashMap<String, (Vec<usize>, usize, usize)>, // name -> (shape, start, len)
}

impl WeightFile {
    fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("读 {path:?} 失败：{e}"))?;
        if bytes.len() < 8 {
            return Err(format!("{path:?} 小于 safetensors 头长度"));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if 8 + header_len > bytes.len() {
            return Err(format!("{path:?} 头长度越界"));
        }
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len])
                .map_err(|e| format!("头 JSON 解析失败：{e}"))?;
        let data_base = 8 + header_len;
        let mut entries = HashMap::new();
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().unwrap_or("");
            if dtype != "F32" {
                return Err(format!("{name} 的 dtype 是 {dtype}，只支持 F32"));
            }
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default();
            let offsets = meta["data_offsets"].as_array().cloned().unwrap_or_default();
            let start = offsets.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            entries.insert(name, (shape, data_base + start, end - start));
        }
        Ok(Self { bytes, entries })
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        let &(ref shape, start, len) = self
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("权重缺 {name}"));
        assert_eq!(shape.iter().product::<usize>(), len / 4, "{name} 长度不符");
        self.bytes[start..start + len]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.entries[name].0.clone()
    }
}

// ---------- 基本算子 ----------

/// torch `nn.GELU()` 默认（approximate='none'）。
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `nn.Linear`：weight 形状 (out, in) 行主序，`y = x·W^T + b`。
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Vec<f32>,
    /// 权重的转置副本 (in_features, out)，预存给矩阵乘用。
    /// torch 的 weight 是 (out, in)，而 `y = x·W^T` 需要 (in, out)。
    weight_t: ndarray::Array2<f32>,
    bias: Vec<f32>,
    out: usize,
    in_features: usize,
}

impl Linear {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let weight = w.f32(&format!("{prefix}.weight"));
        let shape = w.shape(&format!("{prefix}.weight"));
        let bias = w.f32(&format!("{prefix}.bias"));
        let (out, in_features) = (shape[0], shape[1]);
        let weight_t = ndarray::Array2::from_shape_vec((out, in_features), weight.clone())
            .expect("权重整形失败")
            .t()
            .to_owned();
        Self { weight, weight_t, bias, out, in_features }
    }

    fn forward_row(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features, "Linear 输入维度不符");
        let mut y = self.bias.clone();
        for (o, row) in self.weight.chunks_exact(self.in_features).enumerate() {
            let mut acc = 0.0f32;
            for (xi, wi) in x.iter().zip(row) {
                acc += xi * wi;
            }
            y[o] += acc;
        }
        y
    }

    /// 整批前向：一次 (rows×in)·(in×out) 矩阵乘，而不是逐行 matvec。
    ///
    /// ★原先这里是 `xs.iter().map(forward_row)`——每个 token 都把整块权重
    ///   （768×768×4B ≈ 2.25 MB）重扫一遍。单窗 256 个 token 就要扫 256 遍，
    ///   于是 13 个头合计 2334 ms/窗（编码器只要 10 ms）。
    ///   FLOPs 一模一样，差别在：矩阵乘能走分块＋SIMD，且权重只过一遍 cache。
    ///
    /// ★等价性：浮点加法顺序变了，末位可能有差异，但 (a) 每个输出元素仍是同一串
    ///   乘加（只是累加顺序），(b) 下游判据是 argmax／阈值，实测 argmax 不受影响
    ///   （阶段 4 的 argmax 闸与阶段 5/7 的逐字节闸都在改后跑过）。
    fn forward_rows(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows = xs.len();
        if rows == 0 {
            return Vec::new();
        }
        // 输入拼成 (rows, in_features) 连续缓冲
        let mut a = Vec::with_capacity(rows * self.in_features);
        for x in xs {
            assert_eq!(x.len(), self.in_features, "Linear 输入维度不符");
            a.extend_from_slice(x);
        }
        // (rows, in) · (in, out)：按输出列分块撒 rayon，每线程在自己的连续
        // 小缓冲里做 i-k-c 累加（A 行常驻缓存、权重列片只过一遍内存），算完
        // 一次性写回。★不能直接对 y 的列块视图做——行距是 out，内层循环变成
        // 跨步访问，比朴素 dot 还慢一个数量级。原先 `a.dot(&weight_t)` 每个
        // 输出行都把整块权重重扫一遍：reading 头 768×12907×4B≈40MB × 几百行
        // ≈ 每窗 10GB 级内存流量，是 decode 段的大头。乘加次序 k 升序不变，
        // argmax 不受影响（阶段 4 argmax 闸与阶段 5/7 逐字节闸的改序先例）。
        let out = self.out;
        let inn = self.in_features;
        let w_flat: &[f32] = self.weight_t.as_slice().expect("权重应连续");
        let parallel = rows * out >= 1 << 20;
        let block = out.div_ceil(rayon::current_num_threads().max(1)).max(1);
        let ranges: Vec<(usize, usize)> = if parallel {
            (0..out.div_ceil(block))
                .map(|b| (b * block, ((b + 1) * block).min(out)))
                .collect()
        } else {
            vec![(0, out)]
        };
        let bufs: Vec<Vec<f32>> = ranges
            .par_iter()
            .map(|&(c0, c1)| {
                let bc = c1 - c0;
                let mut ylocal = vec![0.0f32; rows * bc];
                for i in 0..rows {
                    let a_row = &a[i * inn..(i + 1) * inn];
                    let y_row = &mut ylocal[i * bc..(i + 1) * bc];
                    for k in 0..inn {
                        let aik = a_row[k];
                        let w_row = &w_flat[k * out + c0..k * out + c1];
                        for (yv, wv) in y_row.iter_mut().zip(w_row.iter()) {
                            *yv += aik * wv;
                        }
                    }
                }
                ylocal
            })
            .collect();
        let mut y = vec![0.0f32; rows * out];
        for (bi, &(c0, c1)) in ranges.iter().enumerate() {
            let bc = c1 - c0;
            for i in 0..rows {
                y[i * out + c0..i * out + c1].copy_from_slice(&bufs[bi][i * bc..(i + 1) * bc]);
            }
        }
        for row in y.chunks_exact_mut(out) {
            for (o, v) in row.iter_mut().enumerate() {
                *v += self.bias[o];
            }
        }
        y.chunks_exact(out).map(|r| r.to_vec()).collect()
    }
}
/// `SequenceLabelingHead`：Linear(hid→hid) → GELU → Linear(hid→labels)。
#[derive(Debug, Clone)]
pub struct SequenceHead {
    inner: Linear,
    outer: Linear,
}

impl SequenceHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        Self {
            inner: Linear::from_weights(w, &format!("{prefix}.0")),
            outer: Linear::from_weights(w, &format!("{prefix}.3")),
        }
    }

    /// 输入 (rows, hid) → 输出 (rows, labels) 的 logits。
    ///
    /// ★外层也必须走 `forward_rows`：reading 头的 labels 是 12907（读音词表），
    ///   逐行 `forward_row` 等于把「逐 token 的 matvec」原样搬到外层——
    ///   256 行 × 12907 标签的权重各扫一遍，单这一个头就要 1.26 s。
    pub fn logits(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mid = self.inner.forward_rows(hidden);
        let activated: Vec<Vec<f32>> = mid
            .iter()
            .map(|row| row.iter().map(|&v| gelu(v)).collect())
            .collect();
        self.outer.forward_rows(&activated)
    }
}

/// `LoRADelta`：dense_a (hid, rank, labels) ＋ dense_b (rank, hid, labels)。
///
/// einsum 链 `bsh,hil->bsil`（其中 hil = hrl,ril->hil）拆成两步，
/// 避免物化 hid×hid×labels 的中间张量（基本句素性 48 标签时 1.1 GB）。
#[derive(Debug, Clone)]
struct LoraDelta {
    a: Vec<f32>,
    b: Vec<f32>,
    hid: usize,
    rank: usize,
    labels: usize,
}

impl LoraDelta {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let a = w.f32(&format!("{prefix}.dense_a"));
        let b = w.f32(&format!("{prefix}.dense_b"));
        let shape = w.shape(&format!("{prefix}.dense_a"));
        let hid = shape[0];
        let rank = shape[1];
        let labels = shape[2];
        Self { a, b, hid, rank, labels }
    }

    /// `tmp[s, r, l] = Σ_h pooled[s, h]·A[h, r, l]`，返回 (rows×rank×labels) 平坦缓冲。
    fn compress(&self, pooled: &[Vec<f32>]) -> Vec<f32> {
        let rows = pooled.len();
        let mut out = vec![0.0f32; rows * self.rank * self.labels];
        for (s, x) in pooled.iter().enumerate() {
            for (h, &xh) in x.iter().enumerate() {
                if xh == 0.0 {
                    continue;
                }
                let a_base = h * self.rank * self.labels;
                let o_base = s * self.rank * self.labels;
                for i in 0..self.rank * self.labels {
                    out[o_base + i] += xh * self.a[a_base + i];
                }
            }
        }
        out
    }

    /// `out[s, i, l] = Σ_r tmp[s, r, l]·B[r, i, l]`，返回 (rows×hid×labels) 平坦缓冲。
    fn expand(&self, tmp: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * self.hid * self.labels];
        for s in 0..rows {
            for r in 0..self.rank {
                for l in 0..self.labels {
                    let t = tmp[s * self.rank * self.labels + r * self.labels + l];
                    if t == 0.0 {
                        continue;
                    }
                    let b_base = r * self.hid * self.labels;
                    let o_base = s * self.hid * self.labels;
                    // ★只累加同一个标签通道 l 的那一列：B 的形状是 (rank, hid, labels)，
                    //   下标＝r·hid·labels ＋ i·labels ＋ l。写成一整段 hid*labels 的连续累加
                    //   会把 tmp[s,r,l] 摊到**全部**标签上，两个标签张量交叉污染。
                    for i in 0..self.hid {
                        let k = i * self.labels + l;
                        out[o_base + k] += t * self.b[b_base + k];
                    }
                }
            }
        }
        out
    }
}

/// `LoRASequenceMultiLabelingHead`：dense ＋ LoRA delta → GELU → classifier → sigmoid。
#[derive(Debug, Clone)]
pub struct LoraMultiLabelHead {
    dense: Linear,
    delta: LoraDelta,
    classifier_weight: Vec<f32>, // (hid, labels)
    classifier_bias: Vec<f32>,   // (labels,)
    labels: usize,
}

impl LoraMultiLabelHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let dense = Linear::from_weights(w, &format!("{prefix}.dense"));
        let delta = LoraDelta::from_weights(w, &format!("{prefix}.delta"));
        let classifier_weight = w.f32(&format!("{prefix}.classifier_weight"));
        let classifier_bias = w.f32(&format!("{prefix}.classifier_bias"));
        let shape = w.shape(&format!("{prefix}.classifier_weight"));
        let labels = shape[1];
        Self { dense, delta, classifier_weight, classifier_bias, labels }
    }

    /// 输入 (rows, hid) → 输出 (rows, labels) 的 sigmoid 概率。
    ///
    /// ★不物化 `expand` 的 (rows×hid×labels) 中间量。
    ///   基本句素性头 rows=256、hid=768、labels=64 ⇒ 1258 万 float／50 MB，
    ///   而它紧接着就被逐元素读走——写一块再读一块纯属浪费（实测这一头 166 ms）。
    ///   这里改成：对每个标签 l，先把该标签的 (hid) 维 delta 向量取出来，
    ///   逐 h 累加 GELU(dense[h] + delta[h,l])·classifier[h,l]。
    ///   与「先 expand 再逐元素读」的算术完全相同，只是不落那块内存。
    pub fn probabilities(&self, pooled: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows = pooled.len();
        let dense_out = self.dense.forward_rows(pooled);
        let tmp = self.delta.compress(pooled);
        let d = &self.delta;
        let mut out = Vec::with_capacity(rows);
        for s in 0..rows {
            let mut probs = Vec::with_capacity(self.labels);
            let tmp_s = &tmp[s * d.rank * d.labels..(s + 1) * d.rank * d.labels];
            let dense_s = &dense_out[s];
            for l in 0..self.labels {
                // delta[s, h, l] = Σ_r tmp[s, r, l] · B[r, h, l]
                let mut acc = self.classifier_bias[l];
                for (h, &dense_h) in dense_s.iter().enumerate() {
                    let mut delta = 0.0f32;
                    for r in 0..d.rank {
                        delta += tmp_s[r * d.labels + l] * d.b[r * d.hid * d.labels + h * d.labels + l];
                    }
                    let activated = gelu(dense_h + delta);
                    acc += activated * self.classifier_weight[h * self.labels + l];
                }
                probs.push(1.0 / (1.0 + (-acc).exp()));
            }
            out.push(probs);
        }
        out
    }
}

/// `WordSelectionHead` 的投影结果：source/target 线性层各算一次，行按需取。
#[derive(Debug, Clone)]
pub struct SelectionProjection {
    hs: Vec<Vec<f32>>,
    ht: Vec<Vec<f32>>,
    output_weight: Vec<f32>, // (labels, hid) 行主序，无 bias
    /// output_weight 的转置 (hid, labels)，预存给矩阵乘用。
    weight_t: ndarray::Array2<f32>,
    labels: usize,
    hid: usize,
}

impl SelectionProjection {
    /// 第 i 行的 logits：`out[j][l] = Σ_h output_w[l,h]·GELU(hs[i,h] + ht[j,h])`。
    /// 第 i 行的全部候选（(ht.len(), labels)）。
    ///
    /// ★labels=1 的头（dependency）逐列标量最快：矩阵乘要为每次调用建
    ///   (cols, hid) 中间数组，那份开销超过收益（实测 86→123 ms）。
    ///   labels>1 的头（discourse，7 类）走 row_gathered 的矩阵乘路径。
    pub fn row(&self, i: usize) -> Vec<Vec<f32>> {
        if self.labels > 1 {
            let cols: Vec<usize> = (0..self.ht.len()).collect();
            return self.row_gathered(i, &cols);
        }
        let hs_i = &self.hs[i];
        let w = &self.output_weight[..self.hid];
        let mut out = Vec::with_capacity(self.ht.len());
        for tj in &self.ht {
            let mut acc = 0.0f32;
            for h in 0..self.hid {
                acc += w[h] * gelu(hs_i[h] + tj[h]);
            }
            out.push(vec![acc]);
        }
        out
    }

    pub fn rows(&self) -> usize {
        self.ht.len()
    }

    /// 只算 `cols` 里那些目标列：`out[k][l]` 对应 `row(i)[cols[k]][l]`。
    /// 逐元素算术与整行算完完全相同（每个 logit 只依赖 (i, j) 这一对）。
    /// 只算 `cols` 里那些目标列。
    ///
    /// ★三步都可矩阵乘：先按对求 GELU(hs[i]+ht[j]) 得 (cols, hid)，
    ///   再与 output_weightᵀ (hid, labels) 相乘。GELU 的输入**不依赖 label**
    ///   （见 head.py:92-96：hidden 只由 hs/ht 决定），所以每个 (i,j) 对只需
    ///   hid 次 GELU，然后一次矩阵乘出全部 label——原先 labels 在外层、
    ///   hid 在内层的写法把这 hid 次 GELU 重算了 labels 遍。
    pub fn row_gathered(&self, i: usize, cols: &[usize]) -> Vec<Vec<f32>> {
        let rows = cols.len();
        let mut act = ndarray::Array2::<f32>::zeros((rows, self.hid));
        for (k, &j) in cols.iter().enumerate() {
            let tj = &self.ht[j];
            let hs_i = &self.hs[i];
            for h in 0..self.hid {
                act[[k, h]] = gelu(hs_i[h] + tj[h]);
            }
        }
        let out = act.dot(&self.weight_t);
        out.rows().into_iter().map(|r| r.to_vec()).collect()
    }

    /// [`Self::row_gathered`] 的再裁剪：只算 `channels` 里那些标签通道，
    /// `out[k][c]` 对应 `row(i)[cols[k]][channels[c]]`（cohesion 按任务裁剪用）。
    pub fn row_gathered_channels(&self, i: usize, cols: &[usize], channels: &[usize]) -> Vec<Vec<f32>> {
        cols.iter()
            .map(|&j| {
                let tj = &self.ht[j];
                channels
                    .iter()
                    .map(|&l| {
                        let w = &self.output_weight[l * self.hid..(l + 1) * self.hid];
                        let mut acc = 0.0f32;
                        for h in 0..self.hid {
                            acc += w[h] * gelu(self.hs[i][h] + tj[h]);
                        }
                        acc
                    })
                    .collect()
            })
            .collect()
    }
}

/// `WordSelectionHead`：l_source／l_target 两个线性层 ＋ 无 bias 的输出层。
#[derive(Debug, Clone)]
pub struct SelectionHead {
    l_source: Linear,
    l_target: Linear,
    output_weight: Vec<f32>,
    labels: usize,
    hid: usize,
}

impl SelectionHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let l_source = Linear::from_weights(w, &format!("{prefix}.l_source"));
        let l_target = Linear::from_weights(w, &format!("{prefix}.l_target"));
        let output_weight = w.f32(&format!("{prefix}.output_layer.weight"));
        let shape = w.shape(&format!("{prefix}.output_layer.weight"));
        let labels = shape[0];
        let hid = shape[1];
        Self { l_source, l_target, output_weight, labels, hid }
    }

    pub fn project(&self, pooled: &[Vec<f32>]) -> SelectionProjection {
        let hs = self.l_source.forward_rows(pooled);
        let ht = self.l_target.forward_rows(pooled);
        let weight_t = ndarray::Array2::from_shape_vec((self.labels, self.hid), self.output_weight.clone())
            .expect("输出层权重整形失败")
            .t()
            .to_owned();
        SelectionProjection {
            hs,
            ht,
            output_weight: self.output_weight.clone(),
            weight_t,
            labels: self.labels,
            hid: self.hid,
        }
    }
}

/// `LoRARelationWiseWordSelectionHead` 的投影结果（delta 已并入 source/target）。
#[derive(Debug, Clone)]
pub struct RelationWiseProjection {
    source: Vec<f32>, // (rows, rels, hid) 平坦
    target: Vec<f32>,
    classifier: Vec<f32>, // (hid, rels)
    rels: usize,
    hid: usize,
}

impl RelationWiseProjection {
    /// 第 i 行的 logits：`out[j][rel] = Σ_h classifier[h,rel]·GELU(source[i,rel] + target[j,rel])`。
    pub fn row(&self, i: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(self.target.len() / (self.rels * self.hid));
        for j in 0..self.target.len() / (self.rels * self.hid) {
            let mut logits = vec![0.0f32; self.rels];
            for (rel, lg) in logits.iter_mut().enumerate() {
                let si = &self.source[i * self.rels * self.hid + rel * self.hid..][..self.hid];
                let tj = &self.target[j * self.rels * self.hid + rel * self.hid..][..self.hid];
                let mut acc = 0.0f32;
                for h in 0..self.hid {
                    acc += self.classifier[h * self.rels + rel] * gelu(si[h] + tj[h]);
                }
                *lg = acc;
            }
            out.push(logits);
        }
        out
    }

    /// 只算 `cols` 里那些目标列：`out[k][rel]` 对应 `row(i)[cols[k]][rel]`。
    ///
    /// ★GELU 的输入依赖 rel（source 与 target 都带 rel 分量，见 head.py:169-176），
    ///   所以不能把 GELU 提到 rel 循环外面——那样会改变结果。
    pub fn row_gathered(&self, i: usize, cols: &[usize]) -> Vec<Vec<f32>> {
        let n_rows = self.target.len() / (self.rels * self.hid);
        cols.iter()
            .map(|&j| {
                assert!(j < n_rows, "cohesion 目标列 {j} 越界（共 {n_rows}）");
                let mut logits = vec![0.0f32; self.rels];
                for (rel, lg) in logits.iter_mut().enumerate() {
                    let si = &self.source[i * self.rels * self.hid + rel * self.hid..][..self.hid];
                    let tj = &self.target[j * self.rels * self.hid + rel * self.hid..][..self.hid];
                    let mut acc = 0.0f32;
                    for h in 0..self.hid {
                        acc += self.classifier[h * self.rels + rel] * gelu(si[h] + tj[h]);
                    }
                    *lg = acc;
                }
                logits
            })
            .collect()
    }
}

impl RelationWiseProjection {
    /// 批量版 [`Self::row_gathered`]：一次算 `rows` 行 × `cols` 列的全部 logits。
    ///
    /// ★与逐行调用**逐位相同**：每对 (i,j) 仍是
    ///   `Σ_h classifier[h,rel] · gelu(source[i,rel,h] + target[j,rel,h])`，
    ///   同一串乘加、同一顺序。改变的只是外层循环的套法——
    ///   逐行调用要重复 101 次函数调用和 101 次 source 切片，
    ///   而且每行的 j 循环彼此独立、无法合并访存。
    ///
    /// ★这也是 cohesion 剪枝的落点：调用方只传**目标行**与**候选列**
    ///   （B×cand），不再算 256×256 全矩阵。
    pub fn rows_gathered(&self, rows: &[usize], cols: &[usize]) -> Vec<Vec<Vec<f32>>> {
        let n_rows = self.target.len() / (self.rels * self.hid);
        for &i in rows {
            for &j in cols {
                assert!(i < n_rows && j < n_rows, "cohesion 下标越界（行 {i} 列 {j}，共 {n_rows}）");
            }
        }
        let mut out = Vec::with_capacity(rows.len());
        for &i in rows {
            out.push(self.row_gathered(i, cols));
        }
        out
    }
}
/// `LoRARelationWiseWordSelectionHead`（head.py:152-190）。
#[derive(Debug, Clone)]
pub struct LoraRelationWiseHead {
    l_source: Linear,
    l_target: Linear,
    delta_source: LoraDelta,
    delta_target: LoraDelta,
    classifier: Vec<f32>, // (hid, rels)
    rels: usize,
    hid: usize,
}

impl LoraRelationWiseHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let l_source = Linear::from_weights(w, &format!("{prefix}.l_source"));
        let l_target = Linear::from_weights(w, &format!("{prefix}.l_target"));
        let delta_source = LoraDelta::from_weights(w, &format!("{prefix}.delta_source"));
        let delta_target = LoraDelta::from_weights(w, &format!("{prefix}.delta_target"));
        let classifier = w.f32(&format!("{prefix}.classifier"));
        let shape = w.shape(&format!("{prefix}.classifier"));
        let hid = shape[0];
        let rels = shape[1];
        Self { l_source, l_target, delta_source, delta_target, classifier, rels, hid }
    }

    /// `source = h_source + delta_source_out`、`target = h_target + delta_target_out`
    /// （head.py:169-170），返回 (rows, rels, hid) 的平坦缓冲。
    pub fn project(&self, pooled: &[Vec<f32>]) -> RelationWiseProjection {
        let rows = pooled.len();
        let hs = self.l_source.forward_rows(pooled);
        let ht = self.l_target.forward_rows(pooled);
        // delta 的 labels 维在这里是 rels
        let ts = self.delta_source.compress(pooled);
        let dso = self.delta_source.expand(&ts, rows);
        let tt = self.delta_target.compress(pooled);
        let dto = self.delta_target.expand(&tt, rows);
        let mut source = vec![0.0f32; rows * self.rels * self.hid];
        let mut target = vec![0.0f32; rows * self.rels * self.hid];
        for s in 0..rows {
            for rel in 0..self.rels {
                for h in 0..self.hid {
                    source[s * self.rels * self.hid + rel * self.hid + h] =
                        hs[s][h] + dso[s * self.hid * self.rels + h * self.rels + rel];
                    target[s * self.rels * self.hid + rel * self.hid + h] =
                        ht[s][h] + dto[s * self.hid * self.rels + h * self.rels + rel];
                }
            }
        }
        RelationWiseProjection { source, target, classifier: self.classifier.clone(), rels: self.rels, hid: self.hid }
    }
}

/// `SequenceLabelingHead(hidden*2, …)`（dependency_type_parser）：
/// 输入是两个 hid 向量的拼接（head.py:157-159 的 cat）。
#[derive(Debug, Clone)]
pub struct PairSequenceHead {
    inner: Linear,
    outer: Linear,
}

impl PairSequenceHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        Self {
            inner: Linear::from_weights(w, &format!("{prefix}.0")),
            outer: Linear::from_weights(w, &format!("{prefix}.3")),
        }
    }

    /// argmax 标签（`dependency_type_predictions = logits.argmax(dim=3)`）。
    pub fn predict_pair(&self, x: &[f32], y: &[f32]) -> usize {
        let mut concat = Vec::with_capacity(x.len() + y.len());
        concat.extend_from_slice(x);
        concat.extend_from_slice(y);
        let mid = self.inner.forward_row(&concat);
        let activated: Vec<f32> = mid.iter().map(|&v| gelu(v)).collect();
        let logits = self.outer.forward_row(&activated);
        crate::sequence_labeling::argmax(&logits)
    }
}

/// word 模块 13 个头 ＋ CRF 的全部权重。
pub struct WordHeads {
    pub reading: SequenceHead,
    pub pos: SequenceHead,
    pub subpos: SequenceHead,
    pub conjtype: SequenceHead,
    pub conjform: SequenceHead,
    pub word_feature: LoraMultiLabelHead,
    pub ne: SequenceHead,
    pub base_phrase_feature: LoraMultiLabelHead,
    pub dependency: SelectionHead,
    pub dependency_type: PairSequenceHead,
    pub cohesion: LoraRelationWiseHead,
    pub discourse: SelectionHead,
    /// CRF 参数（Viterbi 解码用，解码函数在 `sequence_labeling`）。
    pub crf_start: Vec<f32>,
    pub crf_transitions: Vec<f32>, // (num_tags, num_tags) 行主序
    pub crf_end: Vec<f32>,
    pub num_ne_tags: usize,
}

impl WordHeads {
    pub fn load(path: &Path) -> Result<Self, String> {
        let w = WeightFile::load(path)?;
        let crf_start = w.f32("crf.start_transitions");
        let crf_transitions = w.f32("crf.transitions");
        let crf_end = w.f32("crf.end_transitions");
        let num_ne_tags = w.shape("crf.start_transitions")[0];
        Ok(Self {
            reading: SequenceHead::from_weights(&w, "reading_tagger"),
            pos: SequenceHead::from_weights(&w, "pos_tagger"),
            subpos: SequenceHead::from_weights(&w, "subpos_tagger"),
            conjtype: SequenceHead::from_weights(&w, "conjtype_tagger"),
            conjform: SequenceHead::from_weights(&w, "conjform_tagger"),
            word_feature: LoraMultiLabelHead::from_weights(&w, "word_feature_tagger"),
            ne: SequenceHead::from_weights(&w, "ne_tagger"),
            base_phrase_feature: LoraMultiLabelHead::from_weights(&w, "base_phrase_feature_tagger"),
            dependency: SelectionHead::from_weights(&w, "dependency_parser"),
            dependency_type: PairSequenceHead::from_weights(&w, "dependency_type_parser"),
            cohesion: LoraRelationWiseHead::from_weights(&w, "cohesion_analyzer"),
            discourse: SelectionHead::from_weights(&w, "discourse_relation_analyzer"),
            crf_start,
            crf_transitions,
            crf_end,
            num_ne_tags,
        })
    }

    /// 加载后做一次形状自检（标签数与常量表对齐；对不上说明权重与表不同源）。
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("pos", self.pos.outer.out, POS_TAGS.len()),
            ("subpos", self.subpos.outer.out, SUBPOS_TAGS.len()),
            ("conjtype", self.conjtype.outer.out, CONJTYPE_TAGS.len()),
            ("conjform", self.conjform.outer.out, CONJFORM_TAGS.len()),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!("{name} 头的标签数 {got} 与常量表 {want} 不一致"));
            }
        }
        if self.num_ne_tags != self.ne.outer.out {
            return Err(format!(
                "CRF 标签数 {} 与 NE 头 {} 不一致",
                self.num_ne_tags, self.ne.outer.out
            ));
        }
        Ok(())
    }
}
