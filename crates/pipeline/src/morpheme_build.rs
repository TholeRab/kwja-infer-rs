//! 扁平词素组装：为每个词素产出与 KNP 词素行一一对应的字段。
//!
//! features 是「插入序 + 同名键后写覆盖」的小字典（参考实现的 FeatureDict 语义），
//! 用 `OrderedFeats` 表达。同形词（homograph）按 ops 笛卡尔积展开成 ALT 标签。

use crate::jinf::Jinf;
use crate::jumandic::JumanDic;
use crate::word_tables::{AmbigSurfSpec, WordTables};

/// 素性值：布尔真或冒号后的字符串。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureValue {
    True,
    Text(String),
}

/// 插入序字典：键同名时后写覆盖（不改变位置），其余追加。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrderedFeats {
    items: Vec<(String, FeatureValue)>,
}

impl OrderedFeats {
    pub fn set(&mut self, key: impl Into<String>, value: FeatureValue) {
        let key = key.into();
        if let Some(slot) = self.items.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.items.push((key, value));
        }
    }

    pub fn set_true(&mut self, key: impl Into<String>) {
        self.set(key, FeatureValue::True);
    }

    /// 追加一条（不覆盖同名键——NE 一句可有多条）。
    pub fn push(&mut self, key: impl Into<String>, value: FeatureValue) {
        self.items.push((key.into(), value));
    }

    pub fn contains(&self, key: &str) -> bool {
        self.items.iter().any(|(k, _)| k == key)
    }

    pub fn get(&self, key: &str) -> Option<&FeatureValue> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, FeatureValue)> {
        self.items.iter()
    }
}

/// 一个词素的全部 KNP 字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatMorpheme {
    pub surf: String,
    pub reading: String,
    pub lemma: String,
    pub pos: String,
    pub pos_id: usize,
    pub subpos: String,
    pub subpos_id: usize,
    pub conjtype: String,
    pub conjtype_id: usize,
    pub conjform: String,
    pub conjform_id: usize,
    /// 带引号的语义串（无则空串）。
    pub semantics: String,
    pub features: OrderedFeats,
    /// ALT 标签（同形词）。
    pub homographs: Vec<String>,
}

/// 求 lemma：conjtype 为 `*` 时 lemma=norm；
/// 命中 ambig_surf_specs 时走词典（多 lemma 时把余下的记进 homograph_ops）；
/// 否则 jinf 规则变形回「基本形」，失败时回退 norm 并由调用方记 warning。
#[allow(clippy::too_many_arguments)]
fn get_lemma(
    jumandic: &JumanDic,
    jinf: &Jinf,
    ambig_surf_specs: &[AmbigSurfSpec],
    norm: &str,
    pos: &str,
    subpos: &str,
    conjtype: &str,
    conjform: &str,
    homograph_ops: &mut Vec<HomographOp>,
) -> String {
    if conjtype == "*" {
        return norm.to_string();
    }
    // Python 的 for-else：命中 spec 就 break（无论词典查没查到），跳过 jinf；
    // 循环走完（没有任何 spec 命中）才走 jinf。
    let mut ambiguous_lemma: Option<String> = None;
    for spec in ambig_surf_specs {
        if spec.conjtype != conjtype || spec.conjform != conjform {
            continue;
        }
        // 命中同形 spec：查词典定 lemma
        let entries = jumandic.lookup_by_norm(norm);
        let matched: Vec<&crate::jumandic::JumanEntry> = entries
            .iter()
            .filter(|e| e.pos == pos && e.subpos == subpos && e.conjtype == conjtype)
            .collect();
        if !matched.is_empty() {
            // Python 是 set 去重；这里按首现顺序去重（确定性）。
            let mut lemmas: Vec<String> = Vec::new();
            for e in &matched {
                if !lemmas.contains(&e.lemma) {
                    lemmas.push(e.lemma.clone());
                }
            }
            ambiguous_lemma = Some(lemmas[0].clone());
            if lemmas.len() > 1 {
                homograph_ops.push(HomographOp {
                    kind: HomographKind::Lemma,
                    values: lemmas[1..].to_vec(),
                });
            }
        } else {
            // Python: logger.warning(f"failed to get lemma for {norm}") 后 lemma = norm
            ambiguous_lemma = Some(norm.to_string());
        }
        break;
    }
    if let Some(lemma) = ambiguous_lemma {
        return lemma;
    }
    // 非同形：用活用表变形出 lemma
    match jinf.convert(norm, conjtype, conjform, "基本形") {
        Ok(lemma) => lemma,
        Err(e) => {
            // Python: logger.warning 后 lemma = norm。这里无法打 log，行为一致即可。
            let _ = e;
            norm.to_string()
        }
    }
}

/// 查词典语义串，带引号原样返回。
#[allow(clippy::too_many_arguments)]
fn lookup_semantics(
    jumandic: &JumanDic,
    reading: &str,
    norm: &str,
    pos: &str,
    subpos: &str,
    conjtype: &str,
    canon: Option<&str>,
    homograph_ops: &mut Vec<HomographOp>,
) -> String {
    let quote = |s: &str| -> String {
        if s.is_empty() {
            String::new()
        } else {
            format!("\"{s}\"")
        }
    };
    let matched: Vec<crate::jumandic::JumanEntry> = match canon {
        Some(canon) => jumandic
            .lookup_by_canon(canon)
            .into_iter()
            .filter(|e| {
                e.reading == reading
                    && e.lemma == norm
                    && e.pos == pos
                    && e.subpos == subpos
                    && e.conjtype == conjtype
            })
            .collect(),
        None => jumandic
            .lookup_by_norm(norm)
            .into_iter()
            .filter(|e| {
                e.reading == reading && e.pos == pos && e.subpos == subpos && e.conjtype == conjtype
            })
            .collect(),
    };
    if let Some(entry) = matched.first() {
        let semantics = quote(&entry.semantics);
        if matched.len() > 1 {
            homograph_ops.push(HomographOp {
                kind: HomographKind::Semantics,
                values: matched[1..].iter().map(|e| quote(&e.semantics)).collect(),
            });
        }
        semantics
    } else {
        String::new()
    }
}

#[derive(Debug, Clone)]
pub struct HomographOp {
    pub kind: HomographKind,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomographKind {
    Lemma,
    Semantics,
}

/// 同形词 ops 笛卡尔积展开成 ALT 标签。
#[allow(clippy::too_many_arguments)]
fn expand_homographs(
    surf: &str,
    reading: &str,
    ops: &[HomographOp],
    lemma: &str,
    base_sem: &str,
    pos_id: usize,
    subpos_id: usize,
    conjtype_id: usize,
    conjform_id: usize,
) -> Vec<String> {
    if ops.is_empty() {
        return Vec::new();
    }
    // itertools.product：最右维变化最快。ops 数量只有 1–2，直接迭代乘积。
    let mut out: Vec<String> = Vec::new();
    let total: usize = ops.iter().map(|o| o.values.len()).product();
    for combo_idx in 0..total {
        let mut hl = lemma.to_string();
        let mut hs = base_sem.to_string();
        let mut rest = combo_idx;
        for op in ops {
            let n = op.values.len();
            let j = rest % n;
            rest /= n;
            let v = &op.values[j];
            match op.kind {
                HomographKind::Lemma => hl = v.clone(),
                HomographKind::Semantics => hs = v.clone(),
            }
        }
        let tag = format!(
            "ALT-{surf}-{reading}-{hl}-{pos_id}-{subpos_id}-{conjtype_id}-{conjform_id}-{hs}"
        );
        if !out.contains(&tag) {
            out.push(tag);
        }
    }
    out
}

/// 输入各词素的 surf、norm（即 char 阶段 lemma 字段）、模型读音、四路属性下标，
/// 产出 KNP 词素字段。
pub struct MorphemeBuildInput<'a> {
    pub surfs: &'a [String],
    pub norms: &'a [String],
    pub readings: &'a [String],
    pub pos_idx: &'a [usize],
    pub subpos_idx: &'a [usize],
    pub conjtype_idx: &'a [usize],
    pub conjform_idx: &'a [usize],
}

pub fn build_morphemes(
    jumandic: &JumanDic,
    jinf: &Jinf,
    tables: &WordTables,
    input: &MorphemeBuildInput<'_>,
) -> Vec<FlatMorpheme> {
    let n = input.surfs.len();
    assert_eq!(input.norms.len(), n, "zip(strict=True)");
    assert_eq!(input.readings.len(), n, "zip(strict=True)");
    assert_eq!(input.pos_idx.len(), n, "zip(strict=True)");
    assert_eq!(input.subpos_idx.len(), n, "zip(strict=True)");
    assert_eq!(input.conjtype_idx.len(), n, "zip(strict=True)");
    assert_eq!(input.conjform_idx.len(), n, "zip(strict=True)");

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let surf = &input.surfs[i];
        let norm = &input.norms[i];
        let reading = &input.readings[i];
        let pos = &tables.pos_tags[input.pos_idx[i]];
        let subpos = &tables.subpos_tags[input.subpos_idx[i]];
        let conjtype = &tables.conjtype_tags[input.conjtype_idx[i]];
        let conjform = &tables.conjform_tags[input.conjform_idx[i]];

        let mut homograph_ops: Vec<HomographOp> = Vec::new();
        let lemma = get_lemma(
            jumandic,
            jinf,
            &tables.ambig_surf_specs,
            norm,
            pos,
            subpos,
            conjtype,
            conjform,
            &mut homograph_ops,
        );
        let sem = lookup_semantics(
            jumandic,
            reading,
            norm,
            pos,
            subpos,
            conjtype,
            None,
            &mut homograph_ops,
        );

        let pos_id = tables.pos_tag2pos_id[pos];
        let subpos_id = tables.pos_tag_subpos_tag2subpos_id[pos][subpos];
        let conjtype_id = input.conjtype_idx[i];
        let conjform_id = tables.conjtype_tag_conjform_tag2conjform_id[conjtype][conjform];

        let homographs = expand_homographs(
            surf,
            reading,
            &homograph_ops,
            &lemma,
            &sem,
            pos_id,
            subpos_id,
            conjtype_id,
            conjform_id,
        );

        out.push(FlatMorpheme {
            surf: surf.clone(),
            reading: reading.clone(),
            lemma,
            pos: pos.clone(),
            pos_id,
            subpos: subpos.clone(),
            subpos_id,
            conjtype: conjtype.clone(),
            conjtype_id,
            conjform: conjform.clone(),
            conjform_id,
            semantics: sem,
            features: OrderedFeats::default(),
            homographs,
        });
    }
    out
}
