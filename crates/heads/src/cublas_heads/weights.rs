//! 权重清单解析与整块上传 GPU。

use std::collections::HashMap;
use std::ffi::c_void;

use super::ffi::{FnMalloc, FnMemcpy, H2D, Ptr};
use super::{Manifest, Weight};

pub(super) fn parse_manifest(bytes: &[u8]) -> Result<Manifest, String> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("manifest 解析失败：{e}"))?;
    let gelu = v
        .get("gelu")
        .ok_or_else(|| "manifest 缺 gelu".to_string())?;
    let sqrt2 = gelu
        .get("sqrt2")
        .and_then(|x| x.as_f64())
        .ok_or("manifest 缺 gelu.sqrt2")? as f32;
    let one = gelu
        .get("one")
        .and_then(|x| x.as_f64())
        .ok_or("manifest 缺 gelu.one")? as f32;
    let half = gelu
        .get("half")
        .and_then(|x| x.as_f64())
        .ok_or("manifest 缺 gelu.half")? as f32;
    let total_bytes = v
        .get("total_bytes")
        .and_then(|x| x.as_u64())
        .ok_or("manifest 缺 total_bytes")? as usize;
    let arr = v
        .get("weights")
        .and_then(|x| x.as_array())
        .ok_or("manifest 缺 weights")?;
    let mut weights = Vec::with_capacity(arr.len());
    for w in arr {
        let name = w
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("权重缺 name")?
            .to_string();
        let offset = w
            .get("offset")
            .and_then(|x| x.as_u64())
            .ok_or("权重缺 offset")? as usize;
        let numel = w
            .get("numel")
            .and_then(|x| x.as_u64())
            .ok_or("权重缺 numel")? as usize;
        let dims = w
            .get("dims")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64())
                    .map(|d| d as usize)
                    .collect()
            })
            .ok_or("权重缺 dims")?;
        weights.push((name, offset, numel, dims));
    }
    Ok(Manifest {
        sqrt2,
        one,
        half,
        weights,
        total_bytes,
    })
}

pub(super) fn upload(
    malloc: FnMalloc,
    memcpy: FnMemcpy,
    blob: &[u8],
    manifest: &Manifest,
) -> Result<HashMap<&'static str, Weight>, String> {
    // 权重全部驻留 GPU。
    let mut weights = HashMap::new();
    for (wname, offset, numel, _dims) in &manifest.weights {
        let key: &'static str = match wname.as_str() {
            "reading_w1" => "reading_w1",
            "reading_b1" => "reading_b1",
            "reading_w2" => "reading_w2",
            "reading_b2" => "reading_b2",
            "pos_w1" => "pos_w1",
            "pos_b1" => "pos_b1",
            "pos_w2" => "pos_w2",
            "pos_b2" => "pos_b2",
            "subpos_w1" => "subpos_w1",
            "subpos_b1" => "subpos_b1",
            "subpos_w2" => "subpos_w2",
            "subpos_b2" => "subpos_b2",
            "conjtype_w1" => "conjtype_w1",
            "conjtype_b1" => "conjtype_b1",
            "conjtype_w2" => "conjtype_w2",
            "conjtype_b2" => "conjtype_b2",
            "conjform_w1" => "conjform_w1",
            "conjform_b1" => "conjform_b1",
            "conjform_w2" => "conjform_w2",
            "conjform_b2" => "conjform_b2",
            "wf_w1" => "wf_w1",
            "wf_b1" => "wf_b1",
            "wf_delta" => "wf_delta",
            "wf_clf" => "wf_clf",
            "wf_b2" => "wf_b2",
            "bpf_w1" => "bpf_w1",
            "bpf_b1" => "bpf_b1",
            "bpf_delta" => "bpf_delta",
            "bpf_clf" => "bpf_clf",
            "bpf_b2" => "bpf_b2",
            "ne_w1" => "ne_w1",
            "ne_b1" => "ne_b1",
            "ne_w2" => "ne_w2",
            "ne_b2" => "ne_b2",
            "dep_ws" => "dep_ws",
            "dep_bs" => "dep_bs",
            "dep_wt" => "dep_wt",
            "dep_bt" => "dep_bt",
            "dis_ws" => "dis_ws",
            "dis_bs" => "dis_bs",
            "dis_wt" => "dis_wt",
            "dis_bt" => "dis_bt",
            "coh_ws" => "coh_ws",
            "coh_bs" => "coh_bs",
            "coh_wt" => "coh_wt",
            "coh_bt" => "coh_bt",
            "coh_delta_s" => "coh_delta_s",
            "coh_delta_t" => "coh_delta_t",
            "dt_w0" => "dt_w0",
            "dt_b0" => "dt_b0",
            "dt_w3" => "dt_w3",
            "dt_b3" => "dt_b3",
            other => return Err(format!("manifest 里出现未知权重名：{other}")),
        };
        let bytes = *numel * 4;
        if *offset + bytes > blob.len() {
            return Err(format!("权重 {wname} 越界"));
        }
        let mut dev: Ptr = std::ptr::null_mut();
        unsafe {
            if malloc(&mut dev, bytes) != 0 {
                return Err(format!("权重 {wname} cudaMalloc 失败"));
            }
            if memcpy(dev, blob[*offset..].as_ptr() as *const c_void, bytes, H2D) != 0 {
                return Err(format!("权重 {wname} 上传失败"));
            }
        }
        weights.insert(
            key,
            Weight {
                ptr: dev,
                numel: *numel,
            },
        );
    }

    Ok(weights)
}
