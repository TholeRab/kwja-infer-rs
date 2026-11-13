//! tract 编码器微基准：量单窗前向耗时，并验证多线程并发是否真的摊得开。
//!
//! 用法：cargo run --release -p kwja-encoder --example tract_bench
//! （KWJA_BACKEND=tract 时后端即 tract；本示例只测 tract 路径。）

use kwja_encoder::{make_backend, ParallelEncoder, WordEncoderInput};

fn fake_input() -> WordEncoderInput {
    let seq = 256usize;
    let mut input_ids = Vec::with_capacity(seq);
    let mut attention_mask = Vec::with_capacity(seq);
    for i in 0..seq {
        input_ids.push(((i * 7 + 13) % 30000) as i64);
        attention_mask.push(1);
    }
    WordEncoderInput {
        input_ids,
        attention_mask,
        special_token_indices: (0..7i64).map(|i| 200 + i).collect(),
        batch: 1,
        seq,
    }
}

fn main() {
    let mut backend = make_backend(
        "assets/models/char_dynamic.onnx",
        "assets/models/word_encoder7.onnx",
        kwja_encoder::ort_backend::Device::Cpu,
    )
    .expect("后端创建失败");
    let par = backend.as_parallel().expect("该后端不支持共享调用");
    let input = fake_input();

    // 预热＋建计划
    let t = std::time::Instant::now();
    par.run_word_shared(&input).expect("预热前向失败");
    println!("首窗（含建计划）{:?}", t.elapsed());

    // 串行 4 窗
    let t = std::time::Instant::now();
    for _ in 0..4 {
        par.run_word_shared(&input).expect("串行前向失败");
    }
    println!("串行 4 窗 {:?}", t.elapsed());

    // rayon 并发 12 窗（12 核一档）
    let t = std::time::Instant::now();
    let inputs: Vec<_> = (0..12).map(|_| fake_input()).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = inputs
            .iter()
            .map(|i| scope.spawn(|| par.run_word_shared(i).expect("并发前向失败")))
            .collect();
        for handle in handles {
            handle.join().expect("并发线程失败");
        }
    });
    println!("并发 12 窗 {:?}", t.elapsed());
    println!("线程池 = rayon 全局池，物理核数 = {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
}
