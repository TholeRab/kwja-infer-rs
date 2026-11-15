<a id="zh"></a>
# kwja-infer-rs

[中文](#zh) | [English](#en)

KWJA 的推理侧 Rust 重实现：输入日文原文本，输出 KNP 格式的形态论与依存分析结果。整个流程编译成一个独立 exe，运行时不依赖 Python；编码器由 ONNX Runtime 在 CUDA 上执行，13 个输出头默认走 cuBLAS 直调，分词用 tokenizers crate。

产出与上游 KWJA（Python 实现）绝大多数情况下一致。已知有两类偏差：少数概率值落在 0.5 阈值边界上的判定可能与参考不同；单句超出词阶段窗口（256 减特殊 token）容量时，该句不做形态论分析，原样以 `未定義語` 行保留在产出里。

## 上游项目

本仓是 [KWJA](https://github.com/ku-nlp/kwja)（Kyoto-Waseda Japanese Analyzer）推理侧的重实现。KWJA 由京都大学 ku-nlp 实验室与早稻田大学河原研究室开发，是一个基于预训练语言模型的日语统一分析器，覆盖错字校正、分词、形态论、依存解析、共指与談話関係等任务，输出 [KNP](https://github.com/ku-nlp/knp) 的文節依存格式。

本仓权重转换自 [ku-nlp/kwja-checkpoints](https://huggingface.co/ku-nlp/kwja-checkpoints)（`v2.4`），底座模型为 [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)（含 char-wwm 变体）；KNP 输出行为对齐 `kwja==2.6.0`。素材来源与版本逐项记录在 [assets/SOURCE.md](assets/SOURCE.md)。

## 系统要求

- Windows x64（DLL 文件名与打包脚本均按 Windows 口径）。
- Rust 1.85 或更高（代码使用 edition 2024）。
- NVIDIA GPU 与 CUDA 12.x 驱动；CUDA/cuDNN 运行库已随 `assets/` 自带，无需单独安装。
- 磁盘：`assets/` 约 5.2 GB，另加构建产物。
- 显存需求未做正式实测，欢迎反馈你的运行配置。

## 快速开始

本仓只存代码，`assets/` 不进 git，所以克隆后的第一件事是装配素材，然后构建：

```bash
pip install kwja==2.6.0 onnxruntime-gpu==1.23.2 transformers huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu126

python tools/fetch-assets.py                   # 装配全部素材
python tools/fetch-assets.py --only tokenizers # 只拉分词器（3 MB，不需要 GPU）
python tools/fetch-assets.py --only cuda       # 只拷 CUDA 运行库
```

直连 `huggingface.co` 在部分网络下会被 TLS 拦下（`schannel: CRYPT_E_NO_REVOCATION_CHECK`），可以换镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/fetch-assets.py --only tokenizers
```

```bash
cargo build --release
```

最小运行示例：

```bash
echo '今日は晴れです。' > 输入.txt
target/release/kwja-infer.exe --input 输入.txt > 输出.knp
```

`--device` 可选 `auto`（默认，无 CUDA 时回落 CPU）/ `cuda`（失败即报错，不回落）/ `cpu`；`--model-dir` 可显式指定素材目录（省略时先找当前目录的 `assets/`，再从 exe 所在目录逐级上溯，仍找不到则报错）。

输入是 UTF-8 编码的纯文本。多章输入时每章内容后跟一行单独的 `EOD`，程序分章处理；单章输入可以不加。KNP 结果写 stdout，诊断信息写 stderr。设 `KWJA_TIMING=1` 会在 stderr 打出编码、解码、合并、渲染各段的耗时。

## 输出示例

对「快速开始」里的最小示例实跑（CPU 档，约 1.4 秒完成），stdout 得到：

```
# S-ID:c-0-1 kwja:2.6.0
* 1D
+ 1D <NE:DATE:今日><体言><時間>
今日 きょう 今日 名詞 6 時相名詞 10 * 0 * 0 "代表表記:今日/きょう カテゴリ:時間" <基本句-主辞>
は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は"
* -1D
+ -1D <rel type="ガ" target="今日" sid="c-0-1" id="0"/><rel type="時間" target="今日" sid="c-0-1" id="0"/><用言:判><体言><時制:非過去><レベル:C><状態述語><敬語:丁寧表現><節-区切><節-主辞>
晴れ はれ 晴れ 名詞 6 普通名詞 1 * 0 * 0 "代表表記:晴れ/はれv 自他動詞:他:晴らす/はらす 反義:動詞:曇る/くもる 連用形名詞化:晴れる/はれる" <基本句-主辞><用言表記先頭><用言表記末尾>
です です だ 判定詞 4 * 0 判定詞 25 デス列基本形 27 "代表表記:だ/だ"
。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
EOS
```

格式：`# S-ID:` 注释行标示一句及其编号；`*` 行是文節（bunsetsu）依存行，`+` 行是基本句层；词素行给出表层、读音、原形与词性特征；`EOS` 结束一句。

## 仓库结构

workspace 拆多个 crate，不用 `mod.rs`：

| 路径 | 说明 |
| --- | --- |
| `crates/normalize` | NFKC 与字符变换表归一化 |
| `crates/tokenizer` | 字符侧与词侧分词 |
| `crates/encoder` | 神经网络前向的后端抽象（trait `Encoder`） |
| `crates/heads` | 13 个输出头的 logits 计算与 argmax（CPU 实现），以及它们的 GPU 前向 |
| `crates/pipeline` | 流程编排：字符阶段 → Juman++ 文本 → 分窗 → 词阶段 → KNP 文本 |
| `cli/` | 可执行入口（二进制名 `kwja-infer`） |
| `tests/` | 跨 crate 的集成测试 |
| `tools/` | 素材装配、模型图简化、发行打包 |
| `assets/` | 模型、ORT、CUDA 运行库、词典、分词器；约 5.2 GB，不进 git |
| `_bench/` | 基准与实验产物，不进 git，不参与构建与测试 |

## 架构与前向实现

一次完整分析的前向分三层，各层的执行方式不同。

| 层 | 计算内容 | 权重格式 |
| --- | --- | --- |
| 编码器 | DeBERTa 前向，产出词级与 token 级隐状态 | ONNX（`char_dynamic.onnx` / `word_encoder7.onnx`） |
| 13 个头 | 分类、序列标注、依存、共指、談話関係 | 自定义二进制（`models/cublas/`）或 ONNX 融合图 |
| 分词 | 字符与词的切分 | HuggingFace `tokenizer.json` |

ONNX 是模型文件格式，ONNX Runtime（ORT）是执行该格式的推理引擎，cuBLAS 是 NVIDIA 的矩阵运算库。编码器由 ORT 加载 ONNX 模型、在 CUDA 执行；**13 个头的默认路径不经过 ONNX 也不经过 ORT**，它直接调用 cuBLAS 与自写的 CUDA kernel。

### 编码器

编码器实现由 `KWJA_BACKEND` 选择，两种互斥。

| `KWJA_BACKEND` | 实现 | 设备 |
| --- | --- | --- |
| 不设、`cublas`、`ort`，或其它值 | ONNX Runtime（`crates/encoder/src/ort_backend/`） | 由 `--device` 决定，可 GPU |
| `candle`、`rust` | 纯 Rust candle（`crates/encoder/src/candle_backend/`） | 恒 CPU |

`--device` 只对 ORT 后端有意义。candle 后端在代码里固定为 CPU，并行靠统一的 rayon 线程池，不与 ORT 的 CPU EP 争核。

### 13 个头

头前向用哪条实现，取决于编码器落在哪个设备上。

**编码器在 CUDA 时**，头前向由 `KWJA_BACKEND` 的取值决定：

| `KWJA_BACKEND` | 实现 | 权重 | 覆盖的头 |
| --- | --- | --- | --- |
| 不设或 `cublas` | cuBLAS 直调（`crates/heads/src/cublas_heads/`） | `assets/models/cublas/weights.bin` + `manifest.json` | 全部 13 个 |
| `ort` | ORT 融合图（`crates/heads/src/gpu_heads/`） | `heads_fused.onnx` + `dep_type.onnx` | 融合图 12 个，依存类型 1 张独立图 |

默认是第一条。两条路做的都是矩阵乘与逐元素运算，分类结果一致。

**编码器在 CPU 时**，头前向走 `crates/heads/src/forward/` 的纯 Rust 实现，读 safetensors 权重，按头拆成独立任务交给 rayon 并行。这条路径不依赖 ORT，也不需要 CUDA。

**编码器在 GPU、头在 CPU 是不允许的形态**。它不报错、产出也正确，只是慢，容易被忽略，所以程序在该形态下直接报错退出；测试里由 `assert_device_shape()` 拦截。

## 运行时的 DLL

ONNX Runtime 与 CUDA 运行库都在仓内，启动时按绝对路径加载。

`onnxruntime.dll` 由 `ort::init_from(绝对路径)` 显式加载（`load-dynamic` feature），默认指向 `assets/ort/onnxruntime/capi/onnxruntime.dll`，可用 `KWJA_ORT_DLL` 覆盖。

**不能改成链接期导入加 PATH**：Windows 对静态导入的搜索顺序是 exe 目录 → System32 → PATH，系统里若另有旧版 `onnxruntime.dll`，它会压过 PATH 上的版本，所以必须显式走绝对路径。

CUDA 运行库（cudart64_12、cudnn64_9、cublas64_12 等）在 `assets/cuda/`，启动时按依赖序预载，可用 `KWJA_CUDA_DLL_DIR` 覆盖。ORT 在 Rust 进程内解析这些依赖，不走 `PATH`；`onnxruntime_providers_cuda.dll` 与主 DLL 同目录。

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `KWJA_BACKEND` | 选择编码器实现；编码器在 CUDA 时同时选择头前向实现。取值见「架构与前向实现」 | `cublas` |
| `KWJA_ORT_DLL` | onnxruntime.dll 路径 | `assets/ort/onnxruntime/capi/onnxruntime.dll` |
| `KWJA_CUDA_DLL_DIR` | CUDA 运行库目录 | `assets/cuda` |
| `KWJA_TIMING` | 设任意值则输出分段计时 | 不输出 |
| `KWJA_HF_HUB` | candle 后端用的 HuggingFace 缓存目录 | candle 路线必填 |
| `KWJA_CHAR_CKPT` / `KWJA_WORD_CKPT` | candle 后端的 char／word 权重 safetensors 路径覆盖 | `assets/models/` 内对应文件 |
| `KWJA_DEVICE` | 测试用：`cuda` / `cpu`，决定断言哪种形态 | `cuda` |
| `KWJA_ONNX_DIR` | 测试用：模型目录 | `assets/models` |

## 测试

```bash
cargo test --workspace                                  # 单元与结构测试，约 20 秒
cargo test --release --test aozora_corpus -- --ignored   # 公版语料全 11 章，约 3 分钟，需 GPU
```

测试语料固定为 `crates/pipeline/tests/corpus-aozora/`（青空文库《吾輩は猫である》11 章，公版，可随仓分发）；如需换语料，改 `tests/aozora_corpus.rs` 里的语料路径。

`tests/aozora_corpus.rs` 验的是整条链路跑通、产出结构合法、设备形态正确；它不验与上游 KWJA（Python 实现）的数值等价——**等价性需要自备参考产出对拍，本仓不含基线**。全 11 章那轮默认不跑（`#[ignore]`），因为验的只是「不崩、字段齐」。

## 素材与第三方组件

本仓只存代码。`assets/` 约 5.2 GB 不进 git，每个目录的来源与版本逐项记录在 [assets/SOURCE.md](assets/SOURCE.md)：

| 目录 | 体积 | 来源 |
| --- | --- | --- |
| `models/` | 2.6 GB | HuggingFace `ku-nlp/kwja-checkpoints` 的 `v2.4`，由本项目导出与转换 |
| `cuda/` | 2.1 GB（17 个 DLL） | torch `cu126` wheel 里的 `torch/lib/` |
| `ort/` | 389 MB | `onnxruntime-gpu==1.23.2` wheel |
| `dicts/` | 83 MB | `kwja` 包自带的 `resource/jumandic` |
| `tokenizers/` | 3.2 MB | HuggingFace `ku-nlp/deberta-v2-base-japanese(-char-wwm)` |

装配注意：

- `models` 的导出需要 CUDA GPU：`torch.onnx.export` 走 dynamo 路径，且头模块要求参数与缓冲在同一设备上，纯 CPU 导不出来。
- `cuda` 拷的是当前 Python 环境里的 `torch/lib`，那个环境必须装 cu126 版，装成 CPU 版会拷到一堆用不上的东西。
- **13 头（`heads_fused.onnx` 与 `dep_type.onnx`）需要单独产出**：由上游导出流程生成（12 头融合图加依存类型图，opset 17，batch / token_seq / morph_seq 动态）。`fetch-assets.py` 不代跑这一步；已有产出时直接放进 `assets/models/`。
- `models/` 里的 safetensors 权重与 `cublas/weights.bin` 由上游 ckpt 权重转换、打包而来，转换脚本未随仓发布。

## 许可证

本仓代码以 [MIT](LICENSE) 许可证发布。上游 KWJA 同为 MIT（Copyright (c) 2022 Kyoto University）；随包分发的 JUMAN 词典（jumandic）取自 `kwja` 2.6.0 包，沿用其声明。模型权重转换自 [ku-nlp/kwja-checkpoints](https://huggingface.co/ku-nlp/kwja-checkpoints) `v2.4`，其底座模型 [deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)（含 char-wwm）为 CC-BY-SA-4.0，故权重与分词器素材按 CC-BY-SA-4.0 署名共享；kwja-checkpoints 仓库本身未附许可声明，特此说明。第三方组件：ONNX Runtime（MIT）、PyTorch（BSD-3）、CUDA/cuDNN 运行库（NVIDIA 许可条款）。

## 致谢与引用

KWJA 由京都大学 ku-nlp 实验室与早稻田大学河原研究室开发。若本仓对您的研究有帮助，请同时引用上游论文：

```bibtex
@InProceedings{Ueda2023a,
  author    = {Nobuhiro Ueda and Kazumasa Omura and Takashi Kodama and Hirokazu Kiyomaru and Yugo Murawaki and Daisuke Kawahara and Sadao Kurohashi},
  title     = {KWJA: A Unified Japanese Analyzer Based on Foundation Models},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year      = {2023},
  address   = {Toronto, Canada},
}
```

---

<a id="en"></a>
# kwja-infer-rs

[中文](#zh) | [English](#en)

A Rust reimplementation of the inference side of KWJA: it takes raw Japanese text as input and outputs morphological analysis and dependency parsing results in the KNP format. The whole pipeline compiles into a standalone executable with no Python dependency at runtime; the encoder runs on CUDA via ONNX Runtime, the 13 output heads default to direct cuBLAS calls, and tokenization uses the tokenizers crate.

Output agrees with upstream KWJA (the Python implementation) in the vast majority of cases. Two known deviations: decisions for probabilities exactly on the 0.5 threshold boundary may differ from the reference; when a sentence exceeds the word-stage window capacity (256 minus special tokens), that sentence is not morphologically analyzed and is kept verbatim as an `未定義語` line.

## Upstream project

This repository is a reimplementation of the inference side of [KWJA](https://github.com/ku-nlp/kwja) (Kyoto-Waseda Japanese Analyzer). KWJA is developed by the ku-nlp laboratory at Kyoto University together with the Kawahara laboratory at Waseda University. It is a unified Japanese analyzer built on pre-trained language models, covering typo correction, word segmentation, morphological analysis, dependency parsing, coreference resolution and discourse relation analysis, and it outputs the bunsetsu dependency format of [KNP](https://github.com/ku-nlp/knp).

The weights in this repository are converted from [ku-nlp/kwja-checkpoints](https://huggingface.co/ku-nlp/kwja-checkpoints) (`v2.4`), whose base model is [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese) (including the char-wwm variant); KNP output behavior is aligned with `kwja==2.6.0`. Provenance and versions of all assets are recorded in [assets/SOURCE.md](assets/SOURCE.md).

## System requirements

- Windows x64 (DLL names and packaging scripts assume Windows).
- Rust 1.85 or newer (the code uses edition 2024).
- An NVIDIA GPU with CUDA 12.x drivers; the CUDA/cuDNN runtime libraries ship with `assets/`, so no separate installation is needed.
- Disk: about 5.2 GB for `assets/`, plus build artifacts.
- VRAM requirements have not been formally measured; feedback about your setup is welcome.

## Quick start

This repository contains code only, and `assets/` is not in git, so the first step after cloning is to provision assets, then build:

```bash
pip install kwja==2.6.0 onnxruntime-gpu==1.23.2 transformers huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu126

python tools/fetch-assets.py                   # provision everything
python tools/fetch-assets.py --only tokenizers # tokenizers only (3 MB, no GPU needed)
python tools/fetch-assets.py --only cuda       # copy the CUDA runtime only
```

On some networks, direct access to `huggingface.co` is blocked by TLS inspection (`schannel: CRYPT_E_NO_REVOCATION_CHECK`); a mirror can be used instead:

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/fetch-assets.py --only tokenizers
```

```bash
cargo build --release
```

Minimal run example:

```bash
echo '今日は晴れです。' > 输入.txt
target/release/kwja-infer.exe --input 输入.txt > 输出.knp
```

`--device` accepts `auto` (default; falls back to CPU when CUDA is unavailable) / `cuda` (fails loudly, never silently falls back) / `cpu`; `--model-dir` points at an assets directory explicitly (when omitted, the program looks for `assets/` in the current directory, then walks up from the executable directory, and finally errors out).

Input is UTF-8 plain text. For multi-chapter input, append a line containing only `EOD` after each chapter; a single chapter needs no marker. KNP results go to stdout, diagnostics to stderr. Setting `KWJA_TIMING=1` prints per-stage timings (encode, decode, merge, render) to stderr.

## Output example

Running the minimal example from *Quick start* (CPU shape, about 1.4 s) prints the following to stdout:

```
# S-ID:c-0-1 kwja:2.6.0
* 1D
+ 1D <NE:DATE:今日><体言><時間>
今日 きょう 今日 名詞 6 時相名詞 10 * 0 * 0 "代表表記:今日/きょう カテゴリ:時間" <基本句-主辞>
は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は"
* -1D
+ -1D <rel type="ガ" target="今日" sid="c-0-1" id="0"/><rel type="時間" target="今日" sid="c-0-1" id="0"/><用言:判><体言><時制:非過去><レベル:C><状態述語><敬語:丁寧表現><節-区切><節-主辞>
晴れ はれ 晴れ 名詞 6 普通名詞 1 * 0 * 0 "代表表記:晴れ/はれv 自他動詞:他:晴らす/はらす 反義:動詞:曇る/くもる 連用形名詞化:晴れる/はれる" <基本句-主辞><用言表記先頭><用言表記末尾>
です です だ 判定詞 4 * 0 判定詞 25 デス列基本形 27 "代表表記:だ/だ"
。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
EOS
```

Format: the `# S-ID:` comment line marks a sentence and its id; `*` lines carry bunsetsu dependencies and `+` lines the base-phrase layer; morpheme lines list surface form, reading, lemma and features; `EOS` ends a sentence.

## Repository layout

The workspace is split into several crates without `mod.rs`:

| Path | Purpose |
| --- | --- |
| `crates/normalize` | NFKC and character-mapping normalization |
| `crates/tokenizer` | character-side and word-side tokenization |
| `crates/encoder` | backend abstraction for the neural forward (trait `Encoder`) |
| `crates/heads` | logits computation and argmax for the 13 output heads (CPU implementation), plus their GPU forward |
| `crates/pipeline` | orchestration: character stage → Juman++ text → windowing → word stage → KNP text |
| `cli/` | executable entry point (binary name `kwja-infer`) |
| `tests/` | cross-crate integration tests |
| `tools/` | asset provisioning, model-graph simplification, release packaging |
| `assets/` | models, ORT, CUDA runtime, dictionaries, tokenizers; about 5.2 GB, not in git |
| `_bench/` | benchmarks and experiment artifacts; not in git, excluded from build and tests |

## Architecture and forward pass

One full analysis runs the forward pass in three layers, executed differently.

| Layer | Computation | Weight format |
| --- | --- | --- |
| Encoder | DeBERTa forward, producing token- and word-level hidden states | ONNX (`char_dynamic.onnx` / `word_encoder7.onnx`) |
| 13 heads | classification, sequence labeling, dependency, coreference, discourse relations | custom binary (`models/cublas/`) or fused ONNX graphs |
| Tokenization | character and word segmentation | HuggingFace `tokenizer.json` |

ONNX is a model file format; ONNX Runtime (ORT) is the inference engine that executes it; cuBLAS is the matrix-math library from NVIDIA. The encoder loads the ONNX models via ORT and runs on CUDA; **the default path for the 13 heads goes through neither ONNX nor ORT** — it calls cuBLAS and hand-written CUDA kernels directly.

### Encoder

The encoder implementation is selected by `KWJA_BACKEND`; the two options are mutually exclusive.

| `KWJA_BACKEND` | Implementation | Device |
| --- | --- | --- |
| unset, `cublas`, `ort`, or anything else | ONNX Runtime (`crates/encoder/src/ort_backend/`) | decided by `--device`, GPU possible |
| `candle`, `rust` | pure-Rust candle (`crates/encoder/src/candle_backend/`) | CPU only |

`--device` only matters for the ORT backend. The candle backend is hard-wired to CPU in code; parallelism uses a shared rayon thread pool and does not fight the CPU EP of ORT for cores.

### The 13 heads

Which head implementation is used depends on where the encoder runs.

**Encoder on CUDA** — the head forward is decided by `KWJA_BACKEND`:

| `KWJA_BACKEND` | Implementation | Weights | Heads covered |
| --- | --- | --- | --- |
| unset or `cublas` | direct cuBLAS calls (`crates/heads/src/cublas_heads/`) | `assets/models/cublas/weights.bin` + `manifest.json` | all 13 |
| `ort` | ORT fused graphs (`crates/heads/src/gpu_heads/`) | `heads_fused.onnx` + `dep_type.onnx` | 12 heads fused, dependency types as a separate graph |

The first one is the default. Both paths do the same matrix multiplications and element-wise operations, and their classifications agree.

**Encoder on CPU** — heads run through the pure-Rust implementation in `crates/heads/src/forward/`, reading safetensors weights and splitting per-head tasks across rayon. This path needs neither ORT nor CUDA.

**Encoder on GPU with heads on CPU is not an allowed shape.** It does not error and the output is correct, only slow — easy to miss, so the program exits with an error in that shape; tests enforce it via `assert_device_shape()`.

## Runtime DLLs

Both ONNX Runtime and the CUDA runtime ship inside the repository and are loaded by absolute path at startup.

`onnxruntime.dll` is loaded explicitly via `ort::init_from(absolute path)` (the `load-dynamic` feature); it defaults to `assets/ort/onnxruntime/capi/onnxruntime.dll` and can be overridden with `KWJA_ORT_DLL`.

**Do not switch to import-time linking plus PATH**: Windows searches static imports in the order exe directory → System32 → PATH, so any older `onnxruntime.dll` in System32 would win over PATH; the absolute path is mandatory.

The CUDA runtime libraries (cudart64_12, cudnn64_9, cublas64_12, …) live in `assets/cuda/` and are preloaded in dependency order at startup; override with `KWJA_CUDA_DLL_DIR`. ORT resolves these dependencies inside the Rust process, not via `PATH`; `onnxruntime_providers_cuda.dll` sits next to the main DLL.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `KWJA_BACKEND` | selects the encoder implementation; when the encoder runs on CUDA it also selects the head forward. Values: see *Architecture and forward pass* | `cublas` |
| `KWJA_ORT_DLL` | path to onnxruntime.dll | `assets/ort/onnxruntime/capi/onnxruntime.dll` |
| `KWJA_CUDA_DLL_DIR` | CUDA runtime directory | `assets/cuda` |
| `KWJA_TIMING` | if set to any value, prints per-stage timings | off |
| `KWJA_HF_HUB` | HuggingFace cache directory for the candle backend | required for the candle route |
| `KWJA_CHAR_CKPT` / `KWJA_WORD_CKPT` | override paths of the char/word safetensors weights for the candle backend | corresponding files in `assets/models/` |
| `KWJA_DEVICE` | tests only: `cuda` / `cpu`, selects which shape is asserted | `cuda` |
| `KWJA_ONNX_DIR` | tests only: model directory | `assets/models` |

## Testing

```bash
cargo test --workspace                                  # unit and structural tests, ~20 s
cargo test --release --test aozora_corpus -- --ignored   # public-domain corpus, 11 chapters, ~3 min, GPU required
```

The test corpus is fixed at `crates/pipeline/tests/corpus-aozora/` (11 chapters of *I Am a Cat* from Aozora Bunko — public domain, so it can ship with the repo); to swap the corpus, change the corpus path in `tests/aozora_corpus.rs`.

`tests/aozora_corpus.rs` checks that the whole pipeline runs, that the output structure is legal, and that device shapes are correct. It does not check numerical equivalence with upstream KWJA (the Python implementation) — **equivalence requires reference output you provide yourself; this repository ships no baseline**. The 11-chapter round is `#[ignore]`d by default because it only proves that nothing crashes and all fields are present.

## Assets and third-party components

This repository contains code only. `assets/` totals about 5.2 GB and is not in git; provenance and versions are recorded per directory in [assets/SOURCE.md](assets/SOURCE.md):

| Directory | Size | Source |
| --- | --- | --- |
| `models/` | 2.6 GB | exported and converted by this project from HuggingFace `ku-nlp/kwja-checkpoints` `v2.4` |
| `cuda/` | 2.1 GB (17 DLLs) | `torch/lib/` from the torch `cu126` wheel |
| `ort/` | 389 MB | `onnxruntime-gpu==1.23.2` wheel |
| `dicts/` | 83 MB | `resource/jumandic` bundled with the `kwja` package |
| `tokenizers/` | 3.2 MB | HuggingFace `ku-nlp/deberta-v2-base-japanese(-char-wwm)` |

Provisioning caveats:

- Exporting `models` needs a CUDA GPU: `torch.onnx.export` uses the dynamo path, and the head modules require parameters and buffers on the same device, so pure-CPU export does not work.
- `cuda` copies `torch/lib` from the current Python environment, which must be the cu126 build; a CPU-only torch would copy a pile of unusable files.
- **The 13 heads (`heads_fused.onnx` and `dep_type.onnx`) must be produced separately**: they come from the upstream export flow (a 12-head fused graph plus a dependency-type graph, opset 17, dynamic batch / token_seq / morph_seq). `fetch-assets.py` does not run this step; if you already have the outputs, drop them into `assets/models/`.
- The safetensors weights and `cublas/weights.bin` in `models/` are converted and packed from upstream ckpt weights; the conversion scripts are not part of this repository.

## License

The code of this repository is released under the [MIT](LICENSE) license. Upstream KWJA is likewise MIT (Copyright (c) 2022 Kyoto University); the JUMAN dictionary (jumandic) distributed with the package comes from the `kwja` 2.6.0 package under the same notice. Model weights are converted from [ku-nlp/kwja-checkpoints](https://huggingface.co/ku-nlp/kwja-checkpoints) `v2.4`, whose base model [deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese) (including char-wwm) is CC-BY-SA-4.0, so the weights and tokenizer assets are shared with attribution under CC-BY-SA-4.0; the kwja-checkpoints repository itself carries no license declaration. Third-party components: ONNX Runtime (MIT), PyTorch (BSD-3), CUDA/cuDNN runtime libraries (NVIDIA license terms).

## Acknowledgements and citation

KWJA is developed by the ku-nlp laboratory at Kyoto University and the Kawahara laboratory at Waseda University. If this repository helps your research, please also cite the upstream paper:

```bibtex
@InProceedings{Ueda2023a,
  author    = {Nobuhiro Ueda and Kazumasa Omura and Takashi Kodama and Hirokazu Kiyomaru and Yugo Murawaki and Daisuke Kawahara and Sadao Kurohashi},
  title     = {KWJA: A Unified Japanese Analyzer Based on Foundation Models},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year      = {2023},
  address   = {Toronto, Canada},
}
```
