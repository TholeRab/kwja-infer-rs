//! tract 编码器微基准：量单窗前向耗时，并分别验证两种并行是否真的摊得开。
//!
//! 四种配置：1) 单线程内核串行 4 窗；2) 单线程内核并发 12 窗（窗间并行）；
//! 3) 12 线程内核串行 4 窗（窗内并行，tract multithread-mm）；4) 12 线程内核并发 12 窗。
//! 用法：KWJA_BACKEND=tract cargo run --release -p kwja-encoder --example tract_bench

use kwja_encoder::{ParallelEncoder, WordEncoderInput, make_backend};
use tract_linalg::multithread::{Executor, set_default_executor};

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

fn timed(label: &str, rounds: usize, par: &dyn ParallelEncoder, input: &WordEncoderInput) {
    let t = std::time::Instant::now();
    for _ in 0..rounds {
        par.run_word_shared(input).expect("前向失败");
    }
    println!(
        "{label}：{} 窗共 {:?}（均 {:?}/窗）",
        rounds,
        t.elapsed(),
        t.elapsed() / rounds as u32
    );
}

fn timed_concurrent(
    label: &str,
    threads: usize,
    par: &dyn ParallelEncoder,
    input: &WordEncoderInput,
) {
    let t = std::time::Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| par.run_word_shared(input).expect("并发前向失败")))
            .collect();
        for handle in handles {
            handle.join().expect("并发线程失败");
        }
    });
    println!("{label}：并发 {threads} 窗共 {:?}", t.elapsed());
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

    let t = std::time::Instant::now();
    par.run_word_shared(&input).expect("预热前向失败");
    println!("预热（含建计划）{:?}", t.elapsed());

    timed("① 单线程内核·串行", 4, par, &input);
    timed_concurrent("② 单线程内核", 12, par, &input);

    set_default_executor(Executor::multithread(12));
    timed("③ 12 线程内核·串行", 4, par, &input);
    timed_concurrent("④ 12 线程内核", 12, par, &input);
    println!(
        "物理核数 = {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
}
