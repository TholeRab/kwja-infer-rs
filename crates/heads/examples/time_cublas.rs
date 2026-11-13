//! 测 CublasHeads::load 与单窗 forward_fused 的耗时。
use kwja_heads::cublas_heads::CublasHeads;
use std::time::Instant;

fn rows(seed: usize, r: usize, c: usize) -> Vec<Vec<f32>> {
    let mut s = seed as u64 | 1;
    (0..r)
        .map(|_| {
            (0..c)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s % 2000) as f32 - 1000.0) / 1000.0
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), String> {
    let t0 = Instant::now();
    let cu = CublasHeads::load(std::path::Path::new("assets/models"))?.ok_or("缺 cublas 权重")?;
    let t1 = Instant::now();
    println!("load: {:.3}s", (t1 - t0).as_secs_f32());
    let hidden = rows(1, 224, 768);
    let pooled = rows(2, 176, 768);
    let _ = cu.forward_fused(&hidden, &pooled)?;
    let t2 = Instant::now();
    println!("首窗（含缓冲分配）: {:.3}s", (t2 - t1).as_secs_f32());
    let n = 50;
    let t3 = Instant::now();
    for _ in 0..n {
        let _ = cu.forward_fused(&hidden, &pooled)?;
    }
    let t4 = Instant::now();
    println!(
        "稳态 {} 窗: {:.3}s（{:3} ms/窗）",
        n,
        (t4 - t3).as_secs_f32(),
        (t4 - t3).as_secs_f32() * 1000.0 / n as f32
    );
    Ok(())
}
