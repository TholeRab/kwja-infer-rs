# kwja-infer-rs

KWJA 推理的 Rust 侧实现：ONNX Runtime（CUDA）＋ tokenizers，目标是与 Python 基线
整书逐字节一致。方案见 `../Rust侧实施方案`。

## 结构（workspace 多 crate，无 mod.rs）

- `crates/normalize` — NFKC + TRANSLATION_TABLE 归一化
- `crates/tokenizer` — char 侧与 word 侧分词（tokenizers crate）
- `crates/encoder` — 推理后端抽象（trait Encoder）；当前后端 ORT（`ort_backend/`）
- `crates/heads` — 13 个头（argmax / 掩码表），与后端解耦
- `crates/pipeline` — 流程编排：char 阶段 → Juman++ 文本 → 分窗 → word 阶段 → KNP 文本
- `cli` — `kwja-infer` 可执行入口（`--model-dir` / `--device` / `--input`）

## 构建

```bash
cargo build            # ORT_LIB_PATH 已写死在 .cargo/config.toml
cargo clippy --workspace --all-targets
```

运行时需要 `onnxruntime.dll` 与 `onnxruntime_providers_cuda.dll` 可被加载：
把 `../外部素材目录/ort/1.23.2/onnxruntime/capi` 加进 `PATH`。
