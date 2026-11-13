//! Jinf 活用形变换。数据在 assets/dicts/jinf.json 与 assets/dicts/e_basic_stem_suffix.json。
//!
//! 用法只有一种：把模型给出的活用形还原成基本形（词素行的 lemma 字段）。
//! 语义照抄参考实现：源形后缀剥掉得词干，目标形后缀接回去；
//! 后缀为 `*` 表示不变形（词干原样）；「エ基本形」按词干末音查特殊表。

use std::collections::HashMap;
use std::path::Path;

pub struct Jinf {
    dict: HashMap<String, HashMap<String, String>>,
    e_basic_stem_suffix_dict: HashMap<String, String>,
}

impl Jinf {
    pub fn load(dict_dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<String, String> {
            std::fs::read_to_string(dict_dir.join(name)).map_err(|e| format!("读 {name} 失败：{e}"))
        };
        let dict: HashMap<String, HashMap<String, String>> =
            serde_json::from_str(&read("jinf.json")?)
                .map_err(|e| format!("jinf.json 解析失败：{e}"))?;
        let e_basic_stem_suffix_dict: HashMap<String, String> =
            serde_json::from_str(&read("e_basic_stem_suffix.json")?)
                .map_err(|e| format!("e_basic_stem_suffix.json 解析失败：{e}"))?;
        Ok(Self {
            dict,
            e_basic_stem_suffix_dict,
        })
    }

    /// 把 text 从 source_inf_form 变换到 target_inf_form。
    ///
    /// Python 版在源形不匹配时抛 ValueError，调用方捕获后回退成 norm；
    /// Rust 版用 `Err(String)` 表达同一件事。
    pub fn convert(
        &self,
        text: &str,
        inf_type: &str,
        source_inf_form: &str,
        target_inf_form: &str,
    ) -> Result<String, String> {
        if !self.dict.contains_key(inf_type) {
            return Err(format!("'{inf_type}' is not a valid inflection type"));
        }
        let table = &self.dict[inf_type];
        if !table.contains_key(source_inf_form) {
            return Err(format!(
                "'{source_inf_form}' is not a valid inflection form of '{inf_type}'"
            ));
        }
        if !table.contains_key(target_inf_form) {
            return Err(format!(
                "'{target_inf_form}' is not a valid inflection form of '{inf_type}'"
            ));
        }

        let stem = if source_inf_form == "エ基本形" {
            return Err(format!("unable to identify the stem of '{text}'"));
        } else {
            let inf = &table[source_inf_form];
            if inf == "*" {
                text.to_string()
            } else if text.ends_with(inf.as_str()) {
                text[..text.len() - inf.len()].to_string()
            } else {
                return Err(format!(
                    "'{text}' is not in the '{source_inf_form}' form of '{inf_type}'"
                ));
            }
        };

        // 目标形后缀：エ基本形按词干末音查特殊表并把词干换掉，后缀固定「え」
        let (stem, inf): (String, String) = if target_inf_form == "エ基本形" {
            let mut stem2 = stem;
            if let Some(c) = stem2.chars().last()
                && let Some(suffix) = self.e_basic_stem_suffix_dict.get(&c.to_string())
            {
                stem2.pop();
                stem2.push_str(suffix);
            }
            (stem2, "え".to_string())
        } else {
            (stem, table[target_inf_form].clone())
        };

        if inf == "*" {
            Ok(stem)
        } else {
            Ok(stem + &inf)
        }
    }
}
