//! word 阶段常量表（assets/models/word_tables.json，生成文件）。
//!
//! 标签表（pos/subpos/conjtype/conjform/NE/談話関係/依存类型）与
//! `heads::mask_tables` 里的静态表同源；这里再加载一份是因为
//! morpheme 组装还要 id 映射与 cohesion 配置，静态表只服务解码。

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct WordTables {
    pub pos_tags: Vec<String>,
    pub subpos_tags: Vec<String>,
    pub conjtype_tags: Vec<String>,
    pub conjform_tags: Vec<String>,
    pub pos_tag2pos_id: HashMap<String, usize>,
    pub pos_tag_subpos_tag2subpos_id: HashMap<String, HashMap<String, usize>>,
    pub conjtype_tag_conjform_tag2conjform_id: HashMap<String, HashMap<String, usize>>,
    pub word_features: Vec<String>,
    pub base_phrase_features: Vec<String>,
    pub ne_tags: Vec<String>,
    pub dependency_types: Vec<String>,
    pub dependency_topk: usize,
    pub discourse_relations: Vec<String>,
    pub discourse_threshold: f32,
    pub id2reading: Vec<String>,
    pub masked: f32,
    pub unk: String,
    pub unk_id: usize,
    pub id: String,
    pub id_id: usize,
    pub cohesion_tasks: Vec<String>,
    pub pas_cases: Vec<String>,
    pub br_cases: Vec<String>,
    pub exophora_referents: Vec<String>,
    pub restrict_cohesion_target: bool,
    pub special_tokens: Vec<String>,
    /// 同形词消歧规格：
    /// 命中的 (conjtype, conjform) 组合要走词典查 lemma，而不是 jinf 规则变形。
    pub ambig_surf_specs: Vec<AmbigSurfSpec>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AmbigSurfSpec {
    pub conjtype: String,
    pub conjform: String,
}

impl WordTables {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("读 {path:?} 失败：{e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("{path:?} 解析失败：{e}"))
    }

    /// 数据集里的 special_tokens：滤掉 " "。
    pub fn dataset_special_tokens(&self) -> Vec<&str> {
        self.special_tokens
            .iter()
            .filter(|s| s.as_str() != " ")
            .map(|s| s.as_str())
            .collect()
    }

    /// cohesion 任务对应的判据类名。
    pub fn cohesion_task2name(&self, task: &str) -> &'static str {
        match task {
            "pas_analysis" => "PasExtractor",
            "bridging_reference_resolution" => "BridgingExtractor",
            "coreference_resolution" => "CoreferenceExtractor",
            other => panic!("未知 cohesion 任务：{other}"),
        }
    }

    /// cohesion 任务对应的关系名列表。
    pub fn cohesion_task2rels(&self, task: &str) -> Vec<String> {
        match task {
            "pas_analysis" => self.pas_cases.clone(),
            "bridging_reference_resolution" => self.br_cases.clone(),
            "coreference_resolution" => vec!["=".to_string()],
            other => panic!("未知 cohesion 任务：{other}"),
        }
    }

    /// 模型输出第 rel 通道对应的关系名（按任务顺序拼接）。
    pub fn cohesion_rels(&self) -> Vec<String> {
        let mut rels = Vec::new();
        for task in &self.cohesion_tasks {
            rels.extend(self.cohesion_task2rels(task));
        }
        rels
    }
}
