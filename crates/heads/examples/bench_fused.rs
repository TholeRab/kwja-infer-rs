//! 微基准：融合图多窗拼批的每窗耗时（B=1/2/4/8/16）。
//! 随机输入不代表数值，只代表形状与计算量；跑法：
//!   cargo build --release -p kwja-heads --example bench_fused
//!   PATH=torch/lib… target/release/examples/bench_fused.exe
use kwja_encoder::ort_backend::Device;
use kwja_heads::gpu_heads::GpuHeads;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), String> {
    let models = Path::new("assets/models");
    let g = GpuHeads::load(models, Device::Cuda)?.ok_or("模型缺失")?;
    // 典型 3 章窗：token 数与词素数取中位偏上；最坏档另测
    let (t_n, m_n) = (224usize, 176usize);
    let t: Vec<Vec<f32>> = (0..t_n)
        .map(|i| vec![0.001f32 * (i % 7) as f32; 768])
        .collect();
    let p: Vec<Vec<f32>> = (0..m_n)
        .map(|i| vec![0.002f32 * (i % 5) as f32; 768])
        .collect();
    println!("形状 T={t_n} M={m_n}");
    for b in [1usize, 2, 4, 8, 16] {
        let wins: Vec<(&[Vec<f32>], &[Vec<f32>])> = vec![(&t, &p); b];
        for _ in 0..3 {
            let _ = g.forward_fused_batched(&wins)?;
        }
        let iters = 20;
        let s = Instant::now();
        for _ in 0..iters {
            let _ = g.forward_fused_batched(&wins)?;
        }
        let per = s.elapsed().as_secs_f64() * 1000.0 / iters as f64 / b as f64;
        println!("B={b:>2}  {per:7.2} ms/窗");
    }
    Ok(())
}
