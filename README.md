# kwja-infer-rs

KWJA 推理的 Rust 侧实现：ONNX Runtime（CUDA）＋ tokenizers，目标是与 Python 基线
整书逐字节一致。方案见 `../Rust侧实施方案`。

## 素材（本仓自带，不依赖其它仓）

模型、ORT、tokenizer 全部在 `assets/` 下，**约 1.2 GB，不进 git**，分发时随 exe 打包。
来源、版本、更新步骤见 `assets/SOURCE.md`。

**★ 起本仓不引用 `参考实现仓` 的任何路径。**
验证方式：把那个仓的 `_cache` 整个改名藏起来后，`cargo test --workspace` 仍全过。

## 结构（workspace 多 crate，无 mod.rs）

- `crates/normalize` — NFKC + TRANSLATION_TABLE 归一化
- `crates/tokenizer` — char 侧与 word 侧分词（tokenizers crate）
- `crates/encoder` — 推理后端抽象（trait Encoder）；当前后端 ORT（`ort_backend/`）
- `crates/heads` — 13 个头（argmax / 掩码表），与后端解耦
- `crates/pipeline` — 流程编排：char 阶段 → Juman++ 文本 → 分窗 → word 阶段 → KNP 文本
- `cli` — `kwja-infer` 可执行入口（`--model-dir` / `--device` / `--input`）

## 构建

```bash
cargo build
cargo clippy --workspace --all-targets
```

## 运行时的 DLL

- `onnxruntime.dll` 由 `ort::init_from(绝对路径)` 显式加载（`load-dynamic` feature），默认指向
  `assets/ort/onnxruntime/capi/onnxruntime.dll`（**本仓内**），可用环境变量 `KWJA_ORT_DLL` 覆盖。
  **不能靠链接期导入 + PATH**：本机 `C:\Windows\System32` 有一份旧版 `onnxruntime.dll`（1.17.x），
  Windows 加载器对静态导入的搜索顺序是 exe 目录 → System32 → PATH，System32 永远压过 PATH。
- CUDA 运行时依赖（cudart64_12、cudnn64_9、cublas64_12 等）不在 System32，需在 `PATH` 里加
  `../Python 环境/Lib/site-packages/torch/lib`；`onnxruntime_providers_cuda.dll` 与主 DLL 同目录，可被找到。

## 发行打包

```cmd
tools\build-release.cmd -p kwja-infer     :: 只构建（绕开 C 盘目标目录与 TEMP 链接故障）
tools\package-release.cmd                  :: 构建 + 出包（默认先构建）
tools\package-release.cmd --skip-build     :: 只出包，用现有 target\release\kwja-infer.exe
```

**产物落点**：仓库**外**的 `../release/kwja-infer-<yyyymmdd>-<提交号>/`。

| 为什么这么定 | 说明 |
| --- | --- |
| 不放 `target/` | 那是 cargo 的构建目录，`.gitignore` 已忽略，`cargo clean` 会整目录删掉；构建失败也会弄脏它。发行包放进去等于放在随时会被清掉的地方 |
| 放仓库外 | 一个包 5.1 GB（exe ＋ assets），放仓内会被误提交、也会被 `git status` 一直挂着 |
| 目录名带日期 | 只给提交号时交付方看不出哪个新。日期在前便于排序，提交号在后便于回溯代码 |

**包内容**：`kwja-infer.exe` ＋ `assets/`（models/cuda/ort/dicts/tokenizers）＋ `说明.txt`。
`说明.txt` 由 `tools/写发行说明.py` 从 `tools/发行说明-template.txt` 生成，写入本次的提交号与分支。

**★不要用 PowerShell 写这份说明**：它按系统 ANSI（本机 cp936）读 UTF-8 模板，
中文会被读坏再写坏（实测过一次，整份说明变成乱码）。必须显式 utf-8。
