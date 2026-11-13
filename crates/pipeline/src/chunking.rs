//! 切分与三小步：词素到文节/基本句（概率阈值）、基本句素性、NE、句末边界。

use crate::morpheme_build::{FeatureValue, FlatMorpheme, OrderedFeats};

/// 一句的切分结果：文节列表，每个文节含若干基本句。
#[derive(Debug, Clone, Default)]
pub struct Phrase {
    pub bps: Vec<BasePhrase>,
}

#[derive(Debug, Clone)]
pub struct BasePhrase {
    /// 词素下标（窗内全局下标）。
    pub ms: Vec<usize>,
    /// 主辞词素下标（同上）。
    pub head_gi: usize,
    /// 主辞表层（cohesion 的 rel target 用）。
    pub head_text: String,
    pub feats: OrderedFeats,
}

/// 一句：文节序列。
#[derive(Debug, Clone, Default)]
pub struct Chunk {
    pub phrases: Vec<Phrase>,
}

impl BasePhrase {
    pub fn first_ms(&self) -> usize {
        self.ms[0]
    }

    pub fn last_ms(&self) -> usize {
        *self.ms.last().unwrap()
    }
}

/// WORD_FEATURES 里用到的五个下标（由常量表算出）。
pub struct WordFeatureIdx {
    pub head: usize,
    pub bp_end: usize,
    pub ph_end: usize,
    pub yogen_head: usize,
    pub yogen_tail: usize,
}

impl WordFeatureIdx {
    pub fn new(word_features: &[String]) -> Self {
        let f = |name: &str| {
            word_features
                .iter()
                .position(|s| s == name)
                .unwrap_or_else(|| panic!("WORD_FEATURES 缺 {name}"))
        };
        Self {
            head: f("基本句-主辞"),
            bp_end: f("基本句-区切"),
            ph_end: f("文節-区切"),
            yogen_head: f("用言表記先頭"),
            yogen_tail: f("用言表記末尾"),
        }
    }
}

/// 按 word_feature 概率切文节/基本句。
///
/// word_feature_probs[i] 是第 i 个词素（句内扁平序）的 5 维概率。
pub fn chunk_plain(
    flat_ms: &mut [FlatMorpheme],
    sentence_ranges: &[(usize, usize)],
    word_feature_probs: &[Vec<f32>],
    word_features: &[String],
) -> Vec<Chunk> {
    let idx = WordFeatureIdx::new(word_features);
    let mut out = Vec::with_capacity(sentence_ranges.len());
    for &(start, stop) in sentence_ranges {
        let mut phrases: Vec<Phrase> = Vec::new();
        let mut cur_ph = Phrase::default();
        let mut cur_bp = BasePhrase {
            ms: Vec::new(),
            head_gi: 0,
            head_text: String::new(),
            feats: OrderedFeats::default(),
        };
        for i in start..stop {
            let p = &word_feature_probs[i];
            let f = &mut flat_ms[i].features;
            // 按参考实现的顺序写进 features，输出按插入序拼。
            if p[idx.head] >= 0.5 {
                f.set_true("基本句-主辞");
            }
            if p[idx.yogen_head] >= 0.5 {
                f.set_true("用言表記先頭");
            }
            if p[idx.yogen_tail] >= 0.5 {
                f.set_true("用言表記末尾");
            }
            cur_bp.ms.push(i);
            if p[idx.bp_end] >= 0.5 || p[idx.bp_end] + p[idx.ph_end] >= 1.0 {
                cur_ph.bps.push(cur_bp);
                cur_bp = BasePhrase {
                    ms: Vec::new(),
                    head_gi: 0,
                    head_text: String::new(),
                    feats: OrderedFeats::default(),
                };
            }
            if p[idx.ph_end] >= 0.5 && p[idx.bp_end] + p[idx.ph_end] >= 1.0 {
                phrases.push(cur_ph);
                cur_ph = Phrase::default();
            }
        }
        if !cur_bp.ms.is_empty() {
            cur_ph.bps.push(cur_bp);
        }
        if !cur_ph.bps.is_empty() {
            phrases.push(cur_ph);
        }
        for ph in &mut phrases {
            for bp in &mut ph.bps {
                bp.head_gi = head_of(flat_ms, &bp.ms);
                // head_text：基本句主辞的表层文本（cohesion 的 rel target 用）。
                bp.head_text = flat_ms[bp.head_gi].surf.clone();
            }
        }
        out.push(Chunk { phrases });
    }
    out
}

/// 主辞优先级：内容語(0)、準内容語(1)、基本句-主辞(2)，同优先级取先出现者；都无则 ms[0]。
pub fn head_of(flat_ms: &[FlatMorpheme], ms: &[usize]) -> usize {
    const PRIORITY: [(&str, i32); 3] = [("内容語", 0), ("準内容語", 1), ("基本句-主辞", 2)];
    let mut head = ms[0];
    let mut cur = -1i32;
    for &i in ms {
        let f = &flat_ms[i].features;
        if f.is_empty() {
            continue;
        }
        for (feat, pr) in PRIORITY {
            if f.contains(feat) && pr > cur {
                head = i;
                cur = pr;
            }
        }
    }
    head
}

/// BASE_PHRASE_FEATURES 里 節-主辞 的下标（add_bp_features 用）。
pub fn chu_index(base_phrase_features: &[String]) -> usize {
    base_phrase_features
        .iter()
        .position(|s| s == "節-主辞")
        .expect("BASE_PHRASE_FEATURES 缺 節-主辞")
}

/// 基本句素性 + 節-区切/節-主辞 标注。
///
/// bp_features_prob[gi] 以基本句 head 的窗内全局词素下标取行。
/// `bp_head_gi`：该句每个基本句的 head 全局下标（顺序与 chunks 展开序一致）。
pub fn add_bp_features(
    chunks: &mut Chunk,
    bp_features_prob: &[Vec<f32>],
    _bp_head_gi: &[usize],
    base_phrase_features: &[String],
) {
    let i_chu = chu_index(base_phrase_features);
    let n_ph = chunks.phrases.len();
    let mut boundary_feat: Option<String> = None;
    let mut clause_start = 0usize;
    for pi in 0..n_ph {
        // 先铺一般素性（按 BASE_PHRASE_FEATURES 顺序）
        for bi in 0..chunks.phrases[pi].bps.len() {
            let head_gi = chunks.phrases[pi].bps[bi].head_gi;
            let probs = &bp_features_prob[head_gi];
            for (feat, pr) in base_phrase_features.iter().zip(probs) {
                if feat.starts_with("節-区切") && *pr >= 0.5 {
                    boundary_feat = Some(feat.clone());
                } else if feat != "節-主辞" && *pr >= 0.5 {
                    let (k, v) = split_feat(feat);
                    chunks.phrases[pi].bps[bi].feats.set(k, v);
                }
            }
        }
        if pi == n_ph - 1 && boundary_feat.is_none() {
            boundary_feat = Some("節-区切".to_string());
        }
        if let Some(bf) = boundary_feat.take() {
            let (k, v) = split_feat(&bf);
            chunks.phrases[pi].bps.last_mut().unwrap().feats.set(k, v);
            // 在 [clause_start, pi] 这段文节里找 節-主辞 概率最高的基本句
            let mut best: Option<(usize, usize, f32)> = None; // (pi, bi, prob)
            for pj in clause_start..=pi {
                for (bi, bp) in chunks.phrases[pj].bps.iter().enumerate() {
                    let prob = bp_features_prob[bp.head_gi][i_chu];
                    if best.is_none_or(|(_, _, bp_prob)| prob > bp_prob) {
                        best = Some((pj, bi, prob));
                    }
                }
            }
            if let Some((pj, bi, _)) = best {
                chunks.phrases[pj].bps[bi].feats.set_true("節-主辞");
            }
            clause_start = pi + 1;
        }
    }
}

/// '用言:判' 拆成 ('用言', Text('判'))；'体言' 拆成 ('体言', True)。
fn split_feat(feat: &str) -> (String, FeatureValue) {
    match feat.split_once(':') {
        Some((k, v)) => (k.to_string(), FeatureValue::Text(v.to_string())),
        None => (feat.to_string(), FeatureValue::True),
    }
}

/// 逐句扫 NE 标签。
///
/// 返回 (基本句下标, 素性键='NE', 值='类别:文本')；`bp_of_ms` 把窗内词素全局
/// 下标映到基本句序号。
pub fn add_ne(
    flat: &[FlatMorpheme],
    bp_of_ms: &[usize],
    ne_preds: &[usize],
    lo: usize,
    hi: usize,
    ne_tags: &[String],
) -> Vec<(usize, &'static str, String)> {
    let mut out = Vec::new();
    let mut buf: Vec<usize> = Vec::new();
    let mut cat = String::new();

    let flush = |buf: &mut Vec<usize>, cat: &str, out: &mut Vec<(usize, &'static str, String)>| {
        if !buf.is_empty() {
            let text: String = buf.iter().map(|&i| flat[i].surf.as_str()).collect();
            let bpi = bp_of_ms[buf[buf.len() - 1]];
            out.push((bpi, "NE", format!("{cat}:{text}")));
        }
        buf.clear();
    };

    for i in lo..hi {
        let tag = &ne_tags[ne_preds[i]];
        if let Some(category) = tag.strip_prefix("B-") {
            flush(&mut buf, &cat, &mut out);
            cat = category.to_string();
            buf = vec![i];
        } else if let Some(category) = tag.strip_prefix("I-")
            && category == cat
        {
            buf.push(i);
        } else {
            flush(&mut buf, &cat, &mut out);
            cat = String::new();
        }
    }
    flush(&mut buf, &cat, &mut out);
    out
}
