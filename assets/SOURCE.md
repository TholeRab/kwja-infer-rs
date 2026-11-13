# assets/ 素材来源与版本

**为什么有这个文件**： 起本仓**不再引用 `参考实现仓` 的任何路径**，
素材一律拷进本仓 `assets/`。在此之前代码里硬编码了
`外部素材\_cache\...`，跨仓依赖会让本仓无法独立分发。

**验证方式**：把 `外部素材目录` 整个改名藏起来后，
`cargo test --workspace` 仍然全过（25 个测试）。

## 目录

| 目录 | 内容 | 体积 |
| --- | --- | --- |
| `assets/models/` | ONNX 模型 2 个 ＋ 1 个外部权重 | 831 MB |
| `assets/ort/` | onnxruntime-gpu 1.23.2（含 CUDA provider） | 389 MB |
| `assets/tokenizers/` | 两个 HF 模型的 tokenizer 文件 | 3.2 MB |
| `assets/cuda/` | CUDA 12 + cuDNN 9 运行库 14 个 DLL | **2.0 GB** |

**合计约 3.2 GB。全部不进 git**（见 `.gitignore`），
分发时随可执行文件一起打包。

## models/

| 文件 | 大小 | 来源 |
| --- | --- | --- |
| `char_dynamic.onnx` | 404.3 MB | KWJA char 模块导出，batch 固定 1、seq 动态 |
| `word_encoder7.onnx` | 0.1 MB | KWJA word encoder 图（batch/seq 均动态） |
| `word_encoder7.onnx.data` | 426.5 MB | 上者的外部权重 |
| `heads_fused.onnx` | 75.4 MB | word 12 头融合图（`导出Rust侧13头ONNX.py` 导出；opset 17、batch/token_seq/morph_seq 动态；ORT-CPU 对 torch 自检通过） |
| `dep_type.onnx` | 9.5 MB | 依存类型头（同上脚本；输入 (P,1536) 批量动态） |

**★只拷了实际用到的 5 个文件。** 原目录里另有 `_opt.onnx`、`char_full.onnx`、
`word_encoder.onnx`、`word_static256.onnx` 等实验产物（约 1.5 GB），**未拷**。

## ort/

**版本**：`onnxruntime-gpu 1.23.2`（Python wheel 布局，解包即用）。

**为什么锁 1.23.2**：

1. 它是**已经验过、CUDA provider 可用**的那一份；
2. Rust 侧 `ort` crate 用 `api-23` feature，与它对齐；
3. 本机驱动 621.14（CUDA UMD 13.5）其实跑得动更新的版本，
   **不是「跑不了 1.31」**——只是没必要换。

**★必须走 `load-dynamic`，不能静态链接**：
本机 `C:\Windows\System32\onnxruntime.dll` 是 **1.17.1**，
而 Windows 对静态导入的搜索顺序是 exe 目录 → System32 → PATH，
**System32 永远压过 PATH**，会导致「API version 23 not available」。
运行时用 `ort::init_from(绝对路径)` 显式指路才绕得开。

**导入库（.lib / .def / .exp）已删**：走 load-dynamic 用不上，留着会误导后来者走静态链接。
DLL 只导出 4 个符号（`OrtGetApiBase` ＋ 三个 `OrtSessionOptionsAppendExecutionProvider_*`），
这是 ORT 的 C API 设计，不是残缺。

## cuda/

**14 个 DLL，合计约 2.0 GB**，来自 torch 2.14.1+cu126 的 `torch/lib/`。

**为什么必须自带**：ORT 的 CUDA EP 在 Rust 进程里做依赖解析时**不走 PATH**。
不预载时 EP 照样注册成功（`active_providers` 含 CUDA），
**但节点全部静默落 CPU** —— 实测 seq=512 单次 char 前向 **~620 ms**，
预载后 **~25 ms**，**差 25 倍且零报错**。

| DLL | 体积 |
| --- | --- |
| `cublasLt64_12.dll` | 507.2 MB |
| `cusparse64_12.dll` | 274.3 MB |
| `cudnn_adv64_9.dll` | 269.4 MB |
| `cufft64_11.dll` | 264.5 MB |
| `cusolver64_11.dll` | 131.9 MB |
| `cudnn_ops64_9.dll` | 120.6 MB |
| `cublas64_12.dll` | 100.0 MB |
| `curand64_10.dll` | 60.3 MB |
| `cudnn_cnn64_9.dll` | 4.4 MB |
| `cudnn_graph64_9.dll` | 2.3 MB |
| `cudart64_12.dll` | 0.5 MB |
| `cudnn64_9.dll` | 0.3 MB |
| `cudnn_heuristic64_9.dll` | 小 |
| `cudnn_engines_*64_9.dll` | 小 |

**★装载顺序不能乱**：`cudnn_ops64_9.dll` 静态依赖 `cudnn_graph64_9.dll`，
graph 必须先于 ops/cnn/adv 装入。顺序写在 `session.rs` 的 `CUDA_DLLS` 数组里。

**★ 前这里指向仓外** `Python 环境/.../torch/lib`，
现在已拷进本仓，`KWJA_CUDA_DLL_DIR` 仍可覆盖。
验证方式：把那个目录改名藏起来后，`cargo test --workspace` 仍全过。

## tokenizers/

两个 HF 缓存目录，保留 `snapshots/<hash>/` 结构（代码按此结构查找）：

| 目录 | 用于 | 关键文件 |
| --- | --- | --- |
| `models--ku-nlp--deberta-v2-base-japanese-char-wwm` | char 侧 | **只有 `vocab.txt`**（22012 行），无 `tokenizer.json` |
| `models--ku-nlp--deberta-v2-base-japanese` | word 侧 | 有 `tokenizer.json`（vocab 32000） |

**★char 侧没有 `tokenizer.json`** 是个容易踩的坑：
Python 侧 `AutoTokenizer` 返回的是**慢分词器** `BertJapaneseTokenizer`，
Rust 侧是按 transformers 源码逐 id 复刻的（见 `crates/tokenizer/src/char_side.rs`）。

## 更新素材的步骤

```bash
# 1. 从 Python 侧拷（只在需要重新导出模型时）
cp ../外部素材目录/onnx/char_dynamic.onnx        assets/models/
cp ../外部素材目录/onnx/word_encoder7.onnx       assets/models/
cp ../外部素材目录/onnx/word_encoder7.onnx.data  assets/models/

# 2. 跑一遍全量测试，确认没破坏
cargo test --workspace
```

**★拷完要更新本文件的体积与版本记录。**