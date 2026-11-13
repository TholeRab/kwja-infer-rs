//! safetensors 极简读取：8 字节小端头长 + JSON 头 + 数据区，只认 F32。

use std::collections::HashMap;
use std::path::Path;

pub(in crate::forward) struct WeightFile {
    bytes: Vec<u8>,
    entries: HashMap<String, (Vec<usize>, usize, usize)>, // name -> (shape, start, len)
}

impl WeightFile {
    pub(in crate::forward) fn load(path: &Path) -> Result<Self, String> {
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

    pub(in crate::forward) fn f32(&self, name: &str) -> Vec<f32> {
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

    pub(in crate::forward) fn shape(&self, name: &str) -> Vec<usize> {
        self.entries[name].0.clone()
    }
}

// 基本算子
