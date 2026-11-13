//! ORT 输出提取辅助：形状校验、行集拆分与批量切片。

use crate::sequence_labeling::argmax;

/// reading 的 (T,12907) logits 逐行取 argmax（首个最大，与
/// [`crate::sequence_labeling::argmax`] 同口径）。直接在 ORT 的平坦缓冲上扫，
/// 不落成行向量——那份中间量每窗约 13MB，解码侧本就只消费 argmax。
pub(super) fn extract_reading_argmax(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<usize>, String> {
    let value = outputs
        .get(name)
        .ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    let (n, c) = match shape {
        [n, c] => (*n as usize, *c as usize),
        [1, n, c] => (*n as usize, *c as usize),
        other => return Err(format!("{name} 形状应为 [n,c] 或 [1,n,c]，实际 {other:?}")),
    };
    if data.len() != n * c {
        return Err(format!(
            "{name} 数据长度 {} 与形状 ({n},{c}) 不符",
            data.len()
        ));
    }
    Ok(data.chunks_exact(c).map(argmax).collect())
}

/// [n,c] 或 [1,n,c] 的 f32 输出展开成行向量集。
pub(super) fn extract_rows(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let value = outputs
        .get(name)
        .ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    let (n, c) = match shape {
        [n, c] => (*n as usize, *c as usize),
        [1, n, c] => (*n as usize, *c as usize),
        other => return Err(format!("{name} 形状应为 [n,c] 或 [1,n,c]，实际 {other:?}")),
    };
    if data.len() != n * c {
        return Err(format!(
            "{name} 数据长度 {} 与形状 ({n},{c}) 不符",
            data.len()
        ));
    }
    Ok(data.chunks_exact(c).map(<[f32]>::to_vec).collect())
}

/// 4D f32 输出摊平成平坦缓冲（cohesion 的 (1,M,9,768) 用）。
pub(super) fn extract_flat(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Vec<f32>, String> {
    let value = outputs
        .get(name)
        .ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    if shape.len() != 4 || shape[0] != 1 {
        return Err(format!("{name} 形状应为 [1,M,rel,hid]，实际 {shape:?}"));
    }
    Ok(data.to_vec())
}

/// 取一个输出并摊平成 (形状, 数据)，供批量提取共用。
fn fetch_output(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let value = outputs
        .get(name)
        .ok_or_else(|| format!("模型输出里没有 {name}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("输出 {name} 提取失败：{e}"))?;
    let shape = &**shape;
    Ok((shape.iter().map(|d| *d as usize).collect(), data.to_vec()))
}

/// (B, n, c) 取每窗前 counts[k] 行（补零行不取）。
pub(super) fn extract_rows_batched(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    counts: &[usize],
) -> Result<Vec<Vec<Vec<f32>>>, String> {
    let (shape, data) = fetch_output(outputs, name)?;
    if shape.len() != 3 {
        return Err(format!("{name} 形状应为 [B,n,c]，实际 {shape:?}"));
    }
    let (b, n, c) = (shape[0], shape[1], shape[2]);
    if b != counts.len() || data.len() != b * n * c {
        return Err(format!(
            "{name} 形状 ({b},{n},{c}) 与批数 {}／数据量 {} 不符",
            counts.len(),
            data.len()
        ));
    }
    Ok((0..b)
        .map(|k| {
            let base = k * n * c;
            data[base..base + counts[k] * c]
                .chunks_exact(c)
                .map(<[f32]>::to_vec)
                .collect()
        })
        .collect())
}

/// (B, T, c) reading 取每窗前 counts[k] 行的 argmax（首个最大）。
pub(super) fn extract_reading_argmax_batched(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    counts: &[usize],
) -> Result<Vec<Vec<usize>>, String> {
    let (shape, data) = fetch_output(outputs, name)?;
    if shape.len() != 3 {
        return Err(format!("{name} 形状应为 [B,T,c]，实际 {shape:?}"));
    }
    let (b, t, c) = (shape[0], shape[1], shape[2]);
    if b != counts.len() || data.len() != b * t * c {
        return Err(format!(
            "{name} 形状 ({b},{t},{c}) 与批数 {}／数据量 {} 不符",
            counts.len(),
            data.len()
        ));
    }
    Ok((0..b)
        .map(|k| {
            let base = k * t * c;
            data[base..base + counts[k] * c]
                .chunks_exact(c)
                .map(argmax)
                .collect()
        })
        .collect())
}

/// (B, M, 9, 768) cohesion 投影按窗摊平成 (M_k*9*768)。
pub(super) fn extract_coh_batched(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    counts: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let (shape, data) = fetch_output(outputs, name)?;
    if shape.len() != 4 || shape[2] != 9 || shape[3] != 768 {
        return Err(format!("{name} 形状应为 [B,M,9,768]，实际 {shape:?}"));
    }
    let (b, m) = (shape[0], shape[1]);
    let per = 9 * 768;
    if b != counts.len() || data.len() != b * m * per {
        return Err(format!(
            "{name} 形状 ({b},{m}) 与批数 {}／数据量 {} 不符",
            counts.len(),
            data.len()
        ));
    }
    Ok((0..b)
        .map(|k| {
            let base = k * m * per;
            data[base..base + counts[k] * per].to_vec()
        })
        .collect())
}
