//! kwja-infer 命令行入口。
//! 只暴露三个参数：--model-dir / --device(auto|cpu|cuda) / --input；用户不调 batch（事实账 §七）。
//!
//! 设备口径（用户  裁定：**要么全用 CPU，要么全用 GPU，混跑只能是过渡**）：
//! - `cuda`：CUDA EP；起不来直接报错，**绝不静默回落 CPU**；
//! - `cpu`：只装 CPU EP（**纯 CPU 形态**；最终检验＝10 章闸，不跑整书）；
//! - `auto`（默认）：先试 CUDA，起不来自动用纯 CPU，并在 stderr 说明实际用了哪个。

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use anyhow::bail;
use clap::Parser;
use kwja_encoder::ort_backend::Device;
use kwja_encoder::ort_backend::OrtBackend;
use kwja_normalize::nfkc::normalize_text;
use kwja_pipeline::char_stage::CharStage;
use kwja_pipeline::word_predict::WordPredictor;

#[derive(Parser)]
#[command(name = "kwja-infer", about = "KWJA 日文形态论＋KNP 推理（Rust 移植）")]
struct Args {
    /// 素材根目录（assets/：models/ dicts/ tokenizers/）
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// 执行设备：auto | cpu | cuda
    #[arg(long, default_value = "auto")]
    device: String,
    /// 输入文本文件（UTF-8；多章时用 EOD 行分隔，与 Python 侧同一口径；KNP 结果打印到 stdout）
    #[arg(long)]
    input: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("错误：{e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_dir = resolve_model_dir(args.model_dir.clone())?;
    let device = parse_device(&args.device)?;

    // ---- ORT 后端（char/word 共用一个进程级 Environment；设备在此定型）----
    let models = model_dir.join("models");
    let mut backend = OrtBackend::new(
        models.join("char_dynamic.onnx"),
        models.join("word_encoder7.onnx"),
        device,
    )
    .context("ORT 后端创建失败")?;
    let providers = backend.active_providers();
    let has_cuda = providers.iter().any(|p| p.contains("CUDA"));
    if args.device == "cuda" && !has_cuda {
        bail!("要求 CUDA，但 active_providers 不含 CUDA（静默回落 CPU？）：{providers:?}");
    }
    if args.device == "cpu" && has_cuda {
        bail!("要求纯 CPU，active_providers 却含 CUDA：{providers:?}");
    }
    eprintln!(
        "执行设备：{}（active_providers：{providers:?}）",
        if has_cuda { "GPU" } else { "CPU" }
    );

    // ---- 两个阶段：char（RAW → Juman++ 文本）→ word（13 头 → KNP 文本）----
    let char_stage = CharStage::new(char_vocab(&model_dir)?, 512).map_err(anyhow::Error::msg)?;
    let mut predictor = WordPredictor::new(&model_dir, 256, 2).map_err(anyhow::Error::msg)?;
    // 后端与头前向：头前向切 GPU。默认 ORT 融合图（heads_fused.onnx + dep_type.onnx）；
    // KWJA_BACKEND=cublas 切 cuBLAS 直调（models/cublas/ 权重，阶段一/导出cuBLAS权重.py）。
    // 权重缺失或走 CPU 时留 None，头前向回退 Rust CPU 路径，结果口径不变。
    if has_cuda {
        let use_cublas = std::env::var("KWJA_BACKEND").unwrap_or_default() == "cublas";
        if use_cublas {
            match kwja_heads::cublas_heads::CublasHeads::load(&models) {
                Ok(Some(gpu)) => {
                    eprintln!("头前向：cuBLAS 直调（E5，models/cublas 权重）");
                    predictor.set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Cublas(gpu)));
                }
                Ok(None) => eprintln!("头前向：未找到 models/cublas 权重，留在 CPU"),
                Err(e) => return Err(anyhow::anyhow!("加载 cuBLAS 权重失败：{e}")),
            }
        } else {
            match kwja_heads::gpu_heads::GpuHeads::load(&models, device) {
                Ok(Some(gpu)) => {
                    eprintln!("头前向：ORT CUDA（heads_fused.onnx + dep_type.onnx）");
                    predictor.set_gpu_heads(Some(kwja_heads::gpu_backend::GpuHeadsBackend::Ort(gpu)));
                }
                Ok(None) => eprintln!("头前向：未找到 heads_fused.onnx / dep_type.onnx，留在 CPU"),
                Err(e) => return Err(anyhow::anyhow!("加载头 ONNX 失败：{e}")),
            }
        }
    }

    let raw = std::fs::read_to_string(&args.input)
        .with_context(|| format!("读取输入失败：{}", args.input.display()))?;
    // ★按 EOD 切章再逐章归一化（flat_utils.chunk_by_document 同口径：strip 后
    //   等于 "EOD" 的行分块）。Python 侧的写入口径是
    //   `normalize_text(章文本) + "\nEOD\n"`——归一化落在章内，EOD 行不动，
    //   所以这里也必须先切块、后归一化，不能整文件一起归一化。
    let texts: Vec<String> = split_by_eod(&raw).iter().map(|t| normalize_text(t)).collect();

    let started = std::time::Instant::now();
    let jumanpp = char_stage
        .run(&texts, Some("c"), &mut backend)
        .map_err(anyhow::Error::msg)?;
    let knp = predictor
        .predict_text(&mut backend, &jumanpp)
        .map_err(anyhow::Error::msg)?;
    eprintln!(
        "完成：{:?}（设 KWJA_TIMING=1 可看 char/编码/解码/合并分段计时）",
        started.elapsed()
    );
    print!("{knp}");
    Ok(())
}

/// 素材目录：显式 --model-dir → 当前目录 assets/ → exe 上溯 assets/（打包分发）。
fn resolve_model_dir(explicit: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(d) = explicit {
        return Ok(d);
    }
    if Path::new("assets").is_dir() {
        return Ok(PathBuf::from("assets"));
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            let d = ancestor.join("assets");
            if d.is_dir() {
                return Ok(d);
            }
        }
    }
    bail!("找不到素材目录 assets/：请用 --model-dir 指定（含 models/ dicts/ tokenizers/）");
}

/// char 词表：assets/tokenizers 下 char-wwm 的唯一 snapshot。
fn char_vocab(model_dir: &Path) -> anyhow::Result<PathBuf> {
    let snapshots = model_dir
        .join("tokenizers")
        .join("models--ku-nlp--deberta-v2-base-japanese-char-wwm")
        .join("snapshots");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .with_context(|| format!("char-wwm snapshots 目录不存在：{}", snapshots.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    hits.sort();
    if hits.len() != 1 {
        bail!("char-wwm snapshots 不唯一：{hits:?}");
    }
    Ok(hits.remove(0).join("vocab.txt"))
}

/// 按 EOD 行切块（`flat_utils.chunk_by_document` 同口径：
/// `line.strip() == "EOD"` 分块，块尾 rstrip；文件尾不足一块的也算一块）。
fn split_by_eod(text: &str) -> Vec<String> {
    let mut texts: Vec<String> = Vec::new();
    let mut buff = String::new();
    for line in text.split('\n') {
        if line.trim() == "EOD" {
            texts.push(buff.trim_end().to_string());
            buff = String::new();
        } else {
            buff.push_str(line);
            buff.push('\n');
        }
    }
    let rest = buff.trim_end().to_string();
    if !rest.is_empty() {
        texts.push(rest);
    }
    texts
}

fn parse_device(s: &str) -> anyhow::Result<Device> {
    match s {
        "auto" => Ok(Device::Auto),
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => bail!("--device 只接受 auto|cpu|cuda，得到：{other}"),
    }
}
