# kwja-infer-rs

KWJA 的推理侧实现。输入日文原文本，输出 KNP 格式的形态论与依存分析结果。

字符阶段与词阶段的神经网络前向走 ONNX Runtime（CUDA），13 个输出头默认走 cuBLAS 直调；
分词用 tokenizers crate，整个流程编译成一个独立 exe，运行时不依赖 Python。

产出与参考实现绝大多数情况下一致。已知有两类偏差：少数概率值落在 0.5 阈值边界上的判定
可能与参考不同；单句超出词阶段窗口（256 减特殊 token）容量时，该句不做形态论分析，
原样以 `未定義語` 行保留在产出里。

## 结构

workspace 多个 crate，不用 `mod.rs`。

- `crates/normalize` — NFKC 与字符变换表归一化
- `crates/tokenizer` — 字符侧与词侧分词
- `crates/encoder` — 神经网络前向的后端抽象（trait `Encoder`）
- `crates/heads` — 13 个输出头（argmax 与掩码表），以及它们的 GPU 前向
- `crates/pipeline` — 流程编排：字符阶段 → Juman++ 文本 → 分窗 → 词阶段 → KNP 文本
- `cli` — 可执行入口

### 前向的两条路径

**编码器**取哪个实现由 `KWJA_BACKEND` 决定，二者互斥：

| `KWJA_BACKEND` | 编码器实现 | 设备 |
| --- | --- | --- |
| 不设、或 `cublas`、或 `ort` | ONNX Runtime（`ort_backend/`） | 由 `--device` 决定，可 GPU |
| `candle`、`rust` | 纯 Rust candle（`candle_backend/`） | 恒 CPU，靠统一 rayon 线程池并行 |

**13 个头的前向**是另一层选择，只在编码器跑在 CUDA 上时才生效：

| `KWJA_BACKEND` | 头前向实现 |
| --- | --- |
| 不设、或 `cublas` | cuBLAS 直调（`cublas_heads/`），读 `models/cublas/` 权重。**默认** |
| `ort` | ORT 融合图（`gpu_heads/`），读 `heads_fused.onnx` 与 `dep_type.onnx` |
| `candle`、`rust` | 不适用，头在 CPU 上跑 Rust 实现 |

两层由 `gpu_backend.rs` 与 `cli/src/main.rs` 分发。**编码器在 GPU、头在 CPU 是混跑形态，
程序会报错退出**——它不报错时产出也正确，只是慢，容易被忽略。

## 构建与运行

```bash
cargo build --release
cargo clippy --workspace --all-targets
```

素材在仓内 `assets/`，`--model-dir` 可省略。省略时的查找顺序是：当前目录下的 `assets/`，
找不到再从 exe 所在目录逐级上溯找 `assets/`，仍找不到则报错要求显式指定。

```bash
# 单文件输入，KNP 结果打到 stdout
target/release/kwja-infer.exe --input 输入.txt > 输出.knp

# 指定设备：auto（默认，无 CUDA 时回落 CPU）/ cuda（失败即报错，不回落）/ cpu
target/release/kwja-infer.exe --device cuda --input 输入.txt > 输出.knp

# 指定素材目录
target/release/kwja-infer.exe --model-dir D:/kwja-assets --input 输入.txt > 输出.knp
```

输入是 UTF-8 编码的纯文本。多章输入时每章内容后跟一行单独的 `EOD`，分章处理；
单章输入可以不加。KNP 结果写 stdout，诊断信息写 stderr。

设 `KWJA_TIMING=1` 会在 stderr 打出编码、解码、合并、渲染各段的耗时。

## 运行时的 DLL

ONNX Runtime 与 CUDA 运行库都在仓内，启动时按绝对路径加载。

`onnxruntime.dll` 由 `ort::init_from(绝对路径)` 显式加载（`load-dynamic` feature），
默认指向 `assets/ort/onnxruntime/capi/onnxruntime.dll`，可用 `KWJA_ORT_DLL` 覆盖。

**不能改成链接期导入加 PATH**：Windows 对静态导入的搜索顺序是 exe 目录 → System32 → PATH，
系统里若另有旧版 `onnxruntime.dll`，它会压过 PATH 上的版本，所以必须显式走绝对路径。

CUDA 运行库（cudart64_12、cudnn64_9、cublas64_12 等）在 `assets/cuda/`，启动时按依赖序预载，
可用 `KWJA_CUDA_DLL_DIR` 覆盖。ORT 在 Rust 进程内解析这些依赖，不走 `PATH`；
`onnxruntime_providers_cuda.dll` 与主 DLL 同目录。

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `KWJA_BACKEND` | 编码器与头前向的实现，取值见上文「前向的两条路径」 | `cublas` |
| `KWJA_ORT_DLL` | onnxruntime.dll 路径 | `assets/ort/onnxruntime/capi/onnxruntime.dll` |
| `KWJA_CUDA_DLL_DIR` | CUDA 运行库目录 | `assets/cuda` |
| `KWJA_TIMING` | 设任意值则输出分段计时 | 不输出 |
| `KWJA_HF_HUB` | candle 后端用的 HuggingFace 缓存目录 | candle 路线必填 |
| `KWJA_DEVICE` | 测试用：`cuda` / `cpu`，决定断言哪种形态 | `cuda` |
| `本地语料目录` | 测试用：外部语料目录 | 仓内 `corpus-aozora/` |
| `KWJA_ONNX_DIR` | 测试用：模型目录 | `assets/models` |

## 测试

```bash
cargo test --workspace                                  # 单元与结构测试，约 20 秒
cargo test --release --test aozora_corpus -- --ignored   # 公版语料全 11 章，约 3 分钟，需 GPU
```

`tests/aozora_corpus.rs` 用青空文库《吾輩は猫である》验整条链路跑通、产出结构合法、
设备形态正确。它不验与参考实现的数值等价——**等价性需要自备参考产出对拍，本仓不含基线**。
全 11 章那轮默认不跑，因为验的只是「不崩、字段齐」，用一次命令行手动跑同样能看出来。

## 发行打包

```cmd
tools\build-release.cmd -p kwja-infer     :: 只构建，产物固定在 target\
tools\package-release.cmd                  :: 构建 + 出包（默认先构建）
tools\package-release.cmd --skip-build     :: 只出包，用现有 target\release\kwja-infer.exe
```

包落在仓库外的 `../release/kwja-infer-<yyyymmdd>-<提交号>/`。

放仓库外是因为一个包约 3.4 GB（exe 加精简后的素材），放在仓内会被误提交，
也会让 `git status` 一直挂着它。不放 `target/` 是因为那里是 cargo 的构建目录，
`cargo clean` 会整个删掉。目录名日期在前便于排序，提交号在后便于回溯代码。

包内是 `kwja-infer.exe` 加 `assets/` 的五个目录，再加一份 `README.txt`。
`README.txt` 由 `tools/write-release-notes.py` 从 `tools/release-notes-template.txt` 生成，
写入本次的提交号、分支与日期。

**不要用 PowerShell 生成这份说明**：它按系统 ANSI（中文 Windows 为 cp936）读 UTF-8 模板，
中文会被读坏再写坏，必须显式按 utf-8 处理。

## 素材装配

本仓只存代码。`assets/` 约 5.2 GB 不进 git，但每一份都能重新获取。

| 目录 | 体积 | 来源 |
| --- | --- | --- |
| `models/` | 2.6 GB | HuggingFace `ku-nlp/kwja-checkpoints` 的 `v2.4`，由本项目导出 |
| `cuda/` | 2.1 GB | torch `cu126` wheel 里的 `torch/lib/` |
| `ort/` | 389 MB | `onnxruntime-gpu==1.23.2` wheel |
| `dicts/` | 83 MB | `kwja` 包自带的 `resource/jumandic` |
| `tokenizers/` | 3.2 MB | HuggingFace `ku-nlp/deberta-v2-base-japanese(-char-wwm)` |

```bash
pip install kwja==2.6.0 onnxruntime-gpu==1.23.2 transformers huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu126

python tools/fetch-assets.py                   # 全部
python tools/fetch-assets.py --only tokenizers # 只拉分词器，3 MB，不需要 GPU
python tools/fetch-assets.py --only cuda       # 只拷 CUDA 运行库
```

直连 `huggingface.co` 在部分网络下会被 TLS 拦下（`schannel: CRYPT_E_NO_REVOCATION_CHECK`），
可以换镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/fetch-assets.py --only tokenizers
```

有两步容易出问题。一是 `models` 需要 CUDA GPU，因为导出走 `torch.onnx.export` 的 dynamo 路径，
且头模块要求参数与缓冲在同一设备上，纯 CPU 导不出来。二是 `cuda` 拷的是当前 Python 环境里的
`torch/lib`，所以那个环境必须装 cu126 版，装成 CPU 版会拷到一堆用不上的东西。

**13 头（`heads_fused.onnx` 与 `dep_type.onnx`）需要单独产出**。这两个图由上游导出流程生成
（12 头融合图加依赖类型图，opset 17，batch / token_seq / morph_seq 动态）。`fetch-assets.py`
不代跑这一步，因为从 `kwja` 权重重新导出需要上游工程环境。已有产出时直接放进 `assets/models/`。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `assets/` | 模型、ORT、CUDA 运行库、词典、分词器；约 5.2 GB，不进 git |
| `crates/` | 各功能 crate，见上文「结构」 |
| `cli/` | 可执行入口 |
| `tests/` | 跨 crate 的集成测试 |
| `tools/` | 素材装配、模型图简化、发行打包 |
| `_bench/` | 基准与实验产物，不进 git，不参与构建与测试 |
