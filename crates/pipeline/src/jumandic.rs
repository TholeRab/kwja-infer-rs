//! JumanDic（`utils/jumandic.py` 的 Rust 移植）：CDB 词典 ＋ grammar 反查表。
//!
//! 词典文件由导出脚本拷到 `assets/dicts/`：
//! - `jumandic.db`：key = surf，value = 定长 20 字节记录的拼接（`IIHHHHI`：
//!   reading_id, lemma_id, pos_id, subpos_id, conjtype_id, conjform_id, semantics_id）
//! - `jumandic_canon.db`：key = canon（代表表記），value 同上
//! - `grammar.json`：id → 串 的反查表（id2reading / id2lemma / id2pos /
//!   id2subpos / id2conjtype / id2conjform / id2semantics）
//!
//! CDB 是 D. J. Bernstein 的常量库格式：2048 字节头（256 个槽表指针）＋
//! 槽表 ＋ (keylen, vallen, key, value) 记录。这里手写读取，不引依赖。

use std::collections::HashMap;
use std::path::Path;

/// 一条词典记录（Python 版返回 dict，这里用具名结构体）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumanEntry {
    pub reading: String,
    pub lemma: String,
    pub pos: String,
    pub subpos: String,
    pub conjtype: String,
    pub conjform: String,
    pub semantics: String,
}

/// 极简 CDB 读取器：只实现 `get`（本词典每个 key 只有一个 value）。
struct Cdb {
    data: Vec<u8>,
    /// 256 个槽表的 (起始位置, 槽数)
    tables: [(u32, u32); 256],
}

impl Cdb {
    fn open(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("读 {path:?} 失败：{e}"))?;
        if data.len() < 2048 {
            return Err(format!("{path:?} 不是 CDB（头不足 2048 字节）"));
        }
        let mut tables = [(0u32, 0u32); 256];
        for (i, t) in tables.iter_mut().enumerate() {
            let off = i * 8;
            let pos = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let slots = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
            *t = (pos, slots);
        }
        Ok(Self { data, tables })
    }

    /// cdb 的 djb2 变体：h = 5381 起步，`h = h*5 + byte`（mod 2^32）。
    fn hash(key: &[u8]) -> u32 {
        let mut h: u32 = 5381;
        for &b in key {
            h = h.wrapping_mul(5).wrapping_add(b as u32);
        }
        h
    }

    /// 读一条记录：`key_pos` 处是 (klen: u32, vlen: u32, key, value)。
    fn read_record(&self, key_pos: usize) -> Option<(Vec<u8>, Vec<u8>)> {
        let d = &self.data;
        if key_pos + 8 > d.len() {
            return None;
        }
        let klen = u32::from_le_bytes(d[key_pos..key_pos + 4].try_into().unwrap()) as usize;
        let vlen = u32::from_le_bytes(d[key_pos + 4..key_pos + 8].try_into().unwrap()) as usize;
        let key_start = key_pos + 8;
        let val_start = key_start + klen;
        if val_start + vlen > d.len() {
            return None;
        }
        Some((d[key_start..val_start].to_vec(), d[val_start..val_start + vlen].to_vec()))
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let h = Self::hash(key);
        let (table_pos, num_slots) = self.tables[(h % 256) as usize];
        if num_slots == 0 {
            return None;
        }
        // 从 (h >> 8) % num_slots 号槽开始环形扫描
        let mut slot = (h >> 8) % num_slots;
        for _ in 0..num_slots {
            let slot_off = table_pos as usize + slot as usize * 8;
            if slot_off + 8 > self.data.len() {
                return None;
            }
            let slot_hash = u32::from_le_bytes(self.data[slot_off..slot_off + 4].try_into().unwrap());
            let key_pos = u32::from_le_bytes(self.data[slot_off + 4..slot_off + 8].try_into().unwrap());
            if key_pos == 0 {
                // 空槽：cdb 用 key_pos==0 表示空（首记录不可能在 0，头占了 2048）
                return None;
            }
            if slot_hash == h
                && let Some((k, v)) = self.read_record(key_pos as usize)
                && k == key
            {
                return Some(v);
            }
            slot = (slot + 1) % num_slots;
        }
        None
    }
}

struct GrammarTables {
    id2reading: Vec<String>,
    id2lemma: Vec<String>,
    id2pos: Vec<String>,
    id2subpos: Vec<String>,
    id2conjtype: Vec<String>,
    id2conjform: Vec<String>,
    id2semantics: Vec<String>,
}

impl GrammarTables {
    fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("读 {path:?} 失败：{e}"))?;
        let json: HashMap<String, serde_json::Value> =
            serde_json::from_str(&text).map_err(|e| format!("{path:?} 解析失败：{e}"))?;
        let get = |k: &str| -> Result<Vec<String>, String> {
            json.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|x| x.as_str().unwrap_or_default().to_string()).collect())
                .ok_or_else(|| format!("{path:?} 缺 {k}"))
        };
        Ok(Self {
            id2reading: get("id2reading")?,
            id2lemma: get("id2lemma")?,
            id2pos: get("id2pos")?,
            id2subpos: get("id2subpos")?,
            id2conjtype: get("id2conjtype")?,
            id2conjform: get("id2conjform")?,
            id2semantics: get("id2semantics")?,
        })
    }
}

/// JumanDic：两个 CDB ＋ grammar 反查表。
pub struct JumanDic {
    jumandic: Cdb,
    jumandic_canon: Cdb,
    grammar: GrammarTables,
}

/// `struct.unpack("IIHHHHI", ...)` 的记录长：4+4+2+2+2+2+4 = 20 字节。
const RECORD_LEN: usize = 20;

impl JumanDic {
    pub fn load(dic_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            jumandic: Cdb::open(&dic_dir.join("jumandic.db"))?,
            jumandic_canon: Cdb::open(&dic_dir.join("jumandic_canon.db"))?,
            grammar: GrammarTables::load(&dic_dir.join("grammar.json"))?,
        })
    }

    fn unpack_entries(&self, buf: &[u8]) -> Vec<JumanEntry> {
        let g = &self.grammar;
        buf.as_chunks::<RECORD_LEN>()
            .0
            .iter()
            .map(|rec| {
                let reading_id = u32::from_le_bytes(rec[0..4].try_into().unwrap()) as usize;
                let lemma_id = u32::from_le_bytes(rec[4..8].try_into().unwrap()) as usize;
                let pos_id = u16::from_le_bytes(rec[8..10].try_into().unwrap()) as usize;
                let subpos_id = u16::from_le_bytes(rec[10..12].try_into().unwrap()) as usize;
                let conjtype_id = u16::from_le_bytes(rec[12..14].try_into().unwrap()) as usize;
                let conjform_id = u16::from_le_bytes(rec[14..16].try_into().unwrap()) as usize;
                let semantics_id = u32::from_le_bytes(rec[16..20].try_into().unwrap()) as usize;
                JumanEntry {
                    reading: g.id2reading[reading_id].clone(),
                    lemma: g.id2lemma[lemma_id].clone(),
                    pos: g.id2pos[pos_id].clone(),
                    subpos: g.id2subpos[subpos_id].clone(),
                    conjtype: g.id2conjtype[conjtype_id].clone(),
                    conjform: g.id2conjform[conjform_id].clone(),
                    semantics: g.id2semantics[semantics_id].clone(),
                }
            })
            .collect()
    }

    /// `lookup_by_norm`：按 surf 查。
    pub fn lookup_by_norm(&self, surf: &str) -> Vec<JumanEntry> {
        match self.jumandic.get(surf.as_bytes()) {
            Some(buf) => self.unpack_entries(&buf),
            None => Vec::new(),
        }
    }

    /// `lookup_by_canon`：按代表表記查。
    pub fn lookup_by_canon(&self, canon: &str) -> Vec<JumanEntry> {
        match self.jumandic_canon.get(canon.as_bytes()) {
            Some(buf) => self.unpack_entries(&buf),
            None => Vec::new(),
        }
    }
}
