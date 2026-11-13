//! 调试：同一输入分别过 ORT 融合图与 cuBLAS 直调，逐头比最大差与 argmax。
//! 用法：cargo run --release -p kwja-heads --example check_cublas
use kwja_heads::cublas_heads::CublasHeads;
use kwja_heads::gpu_heads::GpuHeads;

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

fn maxdiff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max)
}

fn maxdiff_flat(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max)
}

fn argmax_ne(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn main() -> Result<(), String> {
    let models = std::path::Path::new("assets/models");
    let ort = GpuHeads::load(models, kwja_encoder::ort_backend::Device::Cuda)?.ok_or("缺 onnx")?;
    let cu = CublasHeads::load(models)?.ok_or("缺 cublas 权重")?;
    let hidden = rows(1, 224, 768);
    let pooled = rows(2, 176, 768);
    let a = ort.forward_fused(&hidden, &pooled)?;
    let b = cu.forward_fused(&hidden, &pooled)?;
    println!(
        "reading argmax 不一致数：{}",
        argmax_ne(&a.reading_argmax, &b.reading_argmax)
    );
    println!(
        "pos      maxdiff {:.3e}",
        maxdiff(&a.pos_logits, &b.pos_logits)
    );
    println!(
        "subpos   maxdiff {:.3e}",
        maxdiff(&a.subpos_logits, &b.subpos_logits)
    );
    println!(
        "conjtype maxdiff {:.3e}",
        maxdiff(&a.conjtype_logits, &b.conjtype_logits)
    );
    println!(
        "conjform maxdiff {:.3e}",
        maxdiff(&a.conjform_logits, &b.conjform_logits)
    );
    println!("wf       maxdiff {:.3e}", maxdiff(&a.wf_prob, &b.wf_prob));
    println!("bpf      maxdiff {:.3e}", maxdiff(&a.bpf_prob, &b.bpf_prob));
    println!(
        "ne       maxdiff {:.3e}",
        maxdiff(&a.ne_logits, &b.ne_logits)
    );
    println!("dep_hs   maxdiff {:.3e}", maxdiff(&a.dep_hs, &b.dep_hs));
    println!("dep_ht   maxdiff {:.3e}", maxdiff(&a.dep_ht, &b.dep_ht));
    println!("dis_hs   maxdiff {:.3e}", maxdiff(&a.dis_hs, &b.dis_hs));
    println!("dis_ht   maxdiff {:.3e}", maxdiff(&a.dis_ht, &b.dis_ht));
    println!(
        "coh_s    maxdiff {:.3e}",
        maxdiff_flat(&a.coh_source, &b.coh_source)
    );
    println!("--- 首行前 6 值对照（ort vs cublas）---");
    for (name, x, y) in [
        ("pos", &a.pos_logits, &b.pos_logits),
        ("subpos", &a.subpos_logits, &b.subpos_logits),
        ("conjform", &a.conjform_logits, &b.conjform_logits),
        ("wf", &a.wf_prob, &b.wf_prob),
        ("ne", &a.ne_logits, &b.ne_logits),
    ] {
        let xs: Vec<String> = x[0].iter().take(6).map(|v| format!("{v:.4}")).collect();
        let ys: Vec<String> = y[0].iter().take(6).map(|v| format!("{v:.4}")).collect();
        println!("{name:<8} ort=[{}] cub=[{}]", xs.join(", "), ys.join(", "));
    }
    let ao: Vec<String> = a.dep_hs[0]
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect();
    let bo: Vec<String> = b.dep_hs[0]
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect();
    println!("dep_hs   ort=[{}] cub=[{}]", ao.join(", "), bo.join(", "));
    println!(
        "coh_t    maxdiff {:.3e}",
        maxdiff_flat(&a.coh_target, &b.coh_target)
    );
    Ok(())
}
