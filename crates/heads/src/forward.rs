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
    bias: Vec<f32>,
    out: usize,
    in_features: usize,
}

impl Linear {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let weight = w.f32(&format!("{prefix}.weight"));
        let shape = w.shape(&format!("{prefix}.weight"));
        let bias = w.f32(&format!("{prefix}.bias"));
        Self { weight, bias, out: shape[0], in_features: shape[1] }
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

    fn forward_rows(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter().map(|x| self.forward_row(x)).collect()
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
    pub fn logits(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mid = self.inner.forward_rows(hidden);
        mid.iter()
            .map(|row| {
                let activated: Vec<f32> = row.iter().map(|&v| gelu(v)).collect();
                self.outer.forward_row(&activated)
            })
            .collect()
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
    hid: usize,
    labels: usize,
}

impl LoraMultiLabelHead {
    fn from_weights(w: &WeightFile, prefix: &str) -> Self {
        let dense = Linear::from_weights(w, &format!("{prefix}.dense"));
        let delta = LoraDelta::from_weights(w, &format!("{prefix}.delta"));
        let classifier_weight = w.f32(&format!("{prefix}.classifier_weight"));
        let classifier_bias = w.f32(&format!("{prefix}.classifier_bias"));
        let shape = w.shape(&format!("{prefix}.classifier_weight"));
        let hid = shape[0];
        let labels = shape[1];
        Self { dense, delta, classifier_weight, classifier_bias, hid, labels }
    }

    /// 输入 (rows, hid) → 输出 (rows, labels) 的 sigmoid 概率。
    pub fn probabilities(&self, pooled: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows = pooled.len();
        let dense_out = self.dense.forward_rows(pooled);
        let tmp = self.delta.compress(pooled);
        let ddo = self.delta.expand(&tmp, rows);
        let mut out = Vec::with_capacity(rows);
        for s in 0..rows {
            let mut probs = Vec::with_capacity(self.labels);
            for l in 0..self.labels {
                // hidden[s, h, l] = GELU(dense_out[s, h] + ddo[s, h, l])，再与 classifier[:, l] 内积
                let mut acc = self.classifier_bias[l];
                for h in 0..self.hid {
                    let activated =
                        gelu(dense_out[s][h] + ddo[s * self.hid * self.labels + h * self.labels + l]);
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
    labels: usize,
    hid: usize,
}

impl SelectionProjection {
    /// 第 i 行的 logits：`out[j][l] = Σ_h output_w[l,h]·GELU(hs[i,h] + ht[j,h])`。
    pub fn row(&self, i: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(self.ht.len());
        for tj in &self.ht {
            let mut logits = vec![0.0f32; self.labels];
            for (l, lg) in logits.iter_mut().enumerate() {
                let w = &self.output_weight[l * self.hid..(l + 1) * self.hid];
                let mut acc = 0.0f32;
                for h in 0..self.hid {
                    acc += w[h] * gelu(self.hs[i][h] + tj[h]);
                }
                *lg = acc;
            }
            out.push(logits);
        }
        out
    }

    pub fn rows(&self) -> usize {
        self.ht.len()
    }

    /// 只算 `cols` 里那些目标列：`out[k][l]` 对应 `row(i)[cols[k]][l]`。
    /// 逐元素算术与整行算完完全相同（每个 logit 只依赖 (i, j) 这一对）。
    pub fn row_gathered(&self, i: usize, cols: &[usize]) -> Vec<Vec<f32>> {
        cols.iter()
            .map(|&j| {
                let tj = &self.ht[j];
                let mut logits = vec![0.0f32; self.labels];
                for (l, lg) in logits.iter_mut().enumerate() {
                    let w = &self.output_weight[l * self.hid..(l + 1) * self.hid];
                    let mut acc = 0.0f32;
                    for h in 0..self.hid {
                        acc += w[h] * gelu(self.hs[i][h] + tj[h]);
                    }
                    *lg = acc;
                }
                logits
            })
            .collect()
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
        SelectionProjection { hs, ht, output_weight: self.output_weight.clone(), labels: self.labels, hid: self.hid }
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
