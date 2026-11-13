# -*- coding: utf-8 -*-
"""把发行所需的 assets/ 从零装配出来。

## 为什么需要这个脚本

`assets/` 约 5.2 GB，**全部不进 git**。它里面每一份都能重新获取：

| 目录 | 体积 | 来源 |
| --- | --- | --- |
| `models/` | 2.6 GB | HuggingFace `ku-nlp/kwja-checkpoints` 的 base ckpt，本脚本导出 |
| `cuda/` | 1.6 GB | torch 的 `cu126` wheel 里 `torch/lib/` |
| `ort/` | 389 MB | `onnxruntime-gpu==1.23.2` wheel |
| `dicts/` | 83 MB | KWJA 包自带 `resource/jumandic` |
| `tokenizers/` | 3.2 MB | HuggingFace `ku-nlp/deberta-v2-base-japanese(-char-wwm)` |

所以仓库只存代码，用户跑本脚本就能把素材装回来。

## 用法

    # 需要 Python 3.11+ 与 CUDA 12.x 驱动；建议在独立 venv 里跑
    pip install kwja==2.6.0 torch --index-url https://download.pytorch.org/whl/cu126
    python tools/fetch-assets.py --out assets

## 国内网络

  直连 huggingface.co 在本机被 TLS 拦（schannel 报 CRYPT_E_NO_REVOCATION_CHECK），
  用镜像即可：

      HF_ENDPOINT=https://hf-mirror.com python tools/fetch-assets.py --only tokenizers

  huggingface_hub 认这个环境变量，脚本不用额外改。

## 分步执行（某步失败可单独重跑）

    python tools/fetch-assets.py --only tokenizers   # 只拉分词器（3 MB）
    python tools/fetch-assets.py --only models       # 只导出模型（要 GPU）
    python tools/fetch-assets.py --only cuda         # 只拷 CUDA 运行库

## 依赖关系（写清楚，免得踩空）

- `models` 这一步**必须有 CUDA GPU**：导出走 torch.onnx.export 的 dynamo 路径，
  且头模块要求参数与缓冲全在同一设备。纯 CPU 机器导不出来。
- `cuda` 这一步拷的是**跑本脚本那个 venv 里的 torch/lib**，所以 venv 必须是 cu126 版，
  装成 CPU 版会拷到一堆没用的东西。
- `dicts` 从 `kwja` 包的 `resource/jumandic` 拷；pip 装完就有了。
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
RS_ROOT = HERE.parent          # 仓库根
ASSETS = RS_ROOT / "assets"

# HuggingFace 上的公开仓库与版本（与 KWJA 2.6.0 的 _get_model_version 一致）
CKPT_REPO = "ku-nlp/kwja-checkpoints"
CKPT_REVISION = "v2.4"
CHAR_CKPT = "char_deberta-v2-base-wwm.ckpt"
WORD_CKPT = "word_deberta-v2-base.ckpt"

# 推理时要用的 CUDA DLL（与 crates/encoder/src/ort_backend/session.rs 的 CUDA_DLLS 对齐，
# 另加 cublas 头前向需要的 nvrtc 三个）
CUDA_DLLS = [
    "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
    "cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
    "nvrtc64_120_0.dll", "nvrtc-builtins64_126.dll",
]


def say(msg: str) -> None:
    print(msg, flush=True)


def need_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "需要 CUDA GPU：模型导出走 torch.onnx.export 的 dynamo 路径，纯 CPU 导不出来。"
            "请在有 GPU 的机器上跑，或只跑 --only tokenizers / cuda / dicts。"
        )


def hf_download(repo: str, filename: str, dest_dir: Path, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo, "filename": filename, "local_dir": str(dest_dir)}
    if revision:
        kwargs["revision"] = revision
    return Path(hf_hub_download(**kwargs))


def fetch_tokenizers() -> None:
    """两个 HF 模型的分词器文件，**保留 snapshots/<hash>/ 结构**。

    ★必须保留该结构：Rust 侧按 `tokenizers/models--<org>--<name>/snapshots/*/`
      查找文件（见 crates/tokenizer/tests/ids_match_python.rs:19-27，
      以及 candle_backend.rs 的 snapshot_config）。
      `hf_hub_download(local_dir=...)` 默认落到 `<local_dir>/<文件名>`，不带 snapshots，
      所以这里手动补上这一层。

    ★`spm.model` 不拉：它是 sentencepiece 模型，只有 Python 侧慢分词器用；
      Rust 侧走的是 `vocab.txt`／`tokenizer.json`，从不读 spm（grep 无引用）。
    """
    # ★缓存根必须在 import huggingface_hub **之前**设好：
    #   那个包在导入时就把 HF_HOME 读成模块级常量，之后再 setdefault 无效（实测踩过）。
    #   默认落在 %USERPROFILE%\.cache\huggingface，本机那目录没有写权限（WinError 5）。
    #   这里显式指到 assets/_hfcache，随素材一起删。
    os.environ["HF_HOME"] = str(ASSETS / "_hfcache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    out = ASSETS / "tokenizers"
    pairs = [
        ("ku-nlp/deberta-v2-base-japanese",
         ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]),
        ("ku-nlp/deberta-v2-base-japanese-char-wwm",
         ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]),
    ]
    for repo, files in pairs:
        name = "models--" + repo.replace("/", "--")
        # 取该仓当前 main 的 commit hash，作为 snapshots 下的目录名（与 HF 缓存同一口径）
        rev = HfApi(model_info_timeout=30).model_info(repo).sha if False else HfApi().model_info(repo).sha
        snap = out / name / "snapshots" / rev
        snap.mkdir(parents=True, exist_ok=True)
        (out / name / "refs").mkdir(parents=True, exist_ok=True)
        (out / name / "refs" / "main").write_text(rev, encoding="utf-8")
        got = 0
        for f in files:
            # 镜像偶发 5xx／超时，重试三次；仍失败就如实报出来，不静默跳过
            last = None
            for attempt in range(1, 4):
                try:
                    p = hf_hub_download(repo_id=repo, filename=f, revision=rev)
                    shutil.copyfile(p, snap / f)
                    got += 1
                    last = None
                    break
                except Exception as e:
                    last = e
                    if attempt < 3:
                        time.sleep(2 * attempt)
            if last is not None:
                say("  ★失败 %s/%s：%s: %s" % (repo, f, type(last).__name__, str(last)[:80]))
        say("  %s：%d/%d 个文件 → snapshots/%s" % (repo, got, len(files), rev[:12]))
    say("tokenizers 完成 → %s" % out)


def fetch_ckpt() -> tuple[Path, Path]:
    """从 HF 拉两个 base ckpt（char / word），返回本地路径。"""
    cache = ASSETS / "_ckpt"
    char_p = hf_download(CKPT_REPO, CHAR_CKPT, cache, CKPT_REVISION)
    word_p = hf_download(CKPT_REPO, WORD_CKPT, cache, CKPT_REVISION)
    say("ckpt 就绪：%s, %s" % (char_p.name, word_p.name))
    return char_p, word_p


def export_models(char_ckpt: Path, word_ckpt: Path) -> None:
    """导出 4 个 ONNX：char_dynamic / word_encoder7 / heads_fused / dep_type。

    ## 导的是什么（与 assets/models/ 里现有文件的签名逐项对齐）

    | 产物 | 输入 | 输出 |
    | --- | --- | --- |
    | char_dynamic.onnx | input_ids[1,seq], attention_mask[1,seq] | 3 个标签头 logits |
    | word_encoder7.onnx | input_ids[batch,seq], attention_mask[batch,seq], special_token_indices[batch,special] | last_hidden_state[batch,seq,768] |
    | heads_fused.onnx | hidden, pooled | 12 个头的输出 |
    | dep_type.onnx | pairs[P,1536] | 依存类型 logits |

    ## 为什么 char_dynamic 导出的是「encoder + 3 头」而不是只导 encoder

    char 阶段要的是三个标签头的 logits（分句/分词/NORM_OP），
    把它们一起导出省掉一次 Python 侧的头前向，且减少 Rust 侧要实现的算子。
    输入 batch 维固定 1、seq 动态——char 阶段逐章单条推理，没有批处理。

    ## 与旧导出的差异

    早期 probe 脚本导出的 word 模型带 6 个输入（含 subword_map /
    dependency_mask / cohesion_mask），那是「整个 word 模块」的图；
    现在只需要 **encoder 本体**（13 头在 Rust 侧算），所以是 3 进 1 出。
    本函数按**现行签名**导，不是照抄旧脚本。
    """
    import torch
    from kwja.modules import CharModule, WordModule

    dev = torch.device("cuda")
    out = ASSETS / "models"
    out.mkdir(parents=True, exist_ok=True)

    # ---- char：encoder + 三个标签头 ----
    say("[models] 载入 char ckpt …")
    char_m = CharModule.fast_load_from_checkpoint(
        str(char_ckpt), map_location=dev, accelerator="cuda").eval()

    class CharWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids, am):
            h = self.m.encoder(input_ids=ids, attention_mask=am).last_hidden_state
            return (self.m.sent_segmentation_tagger(h),
                    self.m.word_segmentation_tagger(h),
                    self.m.word_norm_op_tagger(h))

    cw = CharWrap(char_m).eval().to(dev)
    c_ids = torch.ones((1, 32), dtype=torch.long, device=dev)
    c_am = torch.ones((1, 32), dtype=torch.long, device=dev)
    f_char = out / "char_dynamic.onnx"
    torch.onnx.export(
        cw, (c_ids, c_am), str(f_char),
        input_names=["input_ids", "attention_mask"],
        output_names=["sent_segmentation_logits", "word_segmentation_logits",
                      "word_norm_op_logits"],
        dynamic_shapes=({1: "seq"}, {1: "seq"}),
        opset_version=17, dynamo=True, external_data=False,
        verify=False, optimize=False)
    say("  %s  %.1f MB" % (f_char.name, f_char.stat().st_size / 1e6))
    del cw, char_m
    torch.cuda.empty_cache()

    # ---- word：只导 encoder 本体 ----
    say("[models] 载入 word ckpt …")
    word_m = WordModule.fast_load_from_checkpoint(
        str(word_ckpt), map_location=dev, accelerator="cuda").eval()

    class EncWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.e = m.encoder

        def forward(self, ids, am, sti):
            return self.e(input_ids=ids, attention_mask=am,
                          special_token_indices=sti).last_hidden_state

    ew = EncWrap(word_m).eval().to(dev)
    w_ids = torch.ones((2, 32), dtype=torch.long, device=dev)
    w_am = torch.ones((2, 32), dtype=torch.long, device=dev)
    w_sti = torch.zeros((2, 7), dtype=torch.long, device=dev)
    f_word = out / "word_encoder7.onnx"
    torch.onnx.export(
        ew, (w_ids, w_am, w_sti), str(f_word),
        input_names=["input_ids", "attention_mask", "special_token_indices"],
        output_names=["last_hidden_state"],
        dynamic_shapes=({0: "batch", 1: "seq"}, {0: "batch", 1: "seq"}, {0: "batch"}),
        opset_version=17, dynamo=True, external_data=True,
        verify=False, optimize=False)
    size = f_word.stat().st_size + (out / "word_encoder7.onnx.data").stat().st_size \
        if (out / "word_encoder7.onnx.data").exists() else f_word.stat().st_size
    say("  %s  %.1f MB（含外部权重）" % (f_word.name, size / 1e6))
    del ew, word_m
    torch.cuda.empty_cache()

    # ---- 13 头：复用 参考实现仓 的脚本（它自带与 torch 的对拍自检）----
    say("")
    say("[models] 13 头（heads_fused / dep_type）需要另外跑一次：")
    say("    python <参考实现仓>/阶段一/导出Rust侧13头ONNX.py")
    say("  该脚本导出后会自动拷进 assets/models/，并做 ORT-CPU 对拍自检。")
    say("  这里不代跑：那份脚本自己对拍、自己报错，代跑会把两套环境混在一起。")

def fetch_models() -> None:
    """装配 models/：下拉 ckpt，再导出四个 ONNX。"""
    need_cuda()
    char_p, word_p = fetch_ckpt()
    export_models(char_p, word_p)

def fetch_cuda() -> None:
    """从当前 venv 的 torch/lib 拷 CUDA 运行库。"""
    import torch
    lib = Path(torch.__file__).resolve().parent / "lib"
    if not lib.is_dir():
        raise SystemExit("找不到 torch/lib：%s" % lib)
    out = ASSETS / "cuda"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in CUDA_DLLS:
        src = lib / name
        if src.is_file():
            shutil.copyfile(src, out / name)
            n += 1
        else:
            say("  缺 %s（该 venv 的 torch 可能不是 cu126 版）" % name)
    say("cuda 完成：拷了 %d 个 DLL → %s" % (n, out))


def fetch_ort() -> None:
    """解包 onnxruntime-gpu 的 wheel 布局。"""
    try:
        import onnxruntime
    except ImportError:
        raise SystemExit("先 pip install onnxruntime-gpu==1.23.2")
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise SystemExit("这个 onnxruntime 没有 CUDA EP，装成 CPU 版了；重装 onnxruntime-gpu==1.23.2")
    src = Path(onnxruntime.__file__).resolve().parent
    out = ASSETS / "ort" / "onnxruntime"
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("capi",):
        s = src / sub
        if s.is_dir():
            shutil.copytree(s, out / sub, dirs_exist_ok=True)
    say("ort 完成 → %s" % out)


def fetch_dicts() -> None:
    """从 kwja 包拷 jumandic 词典。"""
    try:
        import kwja
    except ImportError:
        raise SystemExit("先 pip install kwja==2.6.0")
    res = Path(kwja.__file__).resolve().parent / "resource" / "jumandic"
    if not res.is_dir():
        raise SystemExit("找不到 %s" % res)
    out = ASSETS / "dicts"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(res, out / "jumandic", dirs_exist_ok=True)
    say("dicts 完成 → %s" % (out / "jumandic"))


STEPS = {
    "tokenizers": fetch_tokenizers,
    "models": fetch_models,
    "cuda": fetch_cuda,
    "ort": fetch_ort,
    "dicts": fetch_dicts,
}


def main() -> int:
    global ASSETS

    ap = argparse.ArgumentParser(description="装配 assets/（全部可重新获取，不进 git）")
    ap.add_argument("--out", default=str(ASSETS), help="assets 目录，默认 <repo>/assets")
    ap.add_argument("--only", choices=sorted(STEPS), action="append",
                    help="只跑指定步骤，可重复")
    args = ap.parse_args()

    ASSETS = Path(args.out)
    ASSETS.mkdir(parents=True, exist_ok=True)

    steps = args.only or list(STEPS)
    for name in steps:
        say("=" * 60)
        say("[%s]" % name)
        try:
            STEPS[name]()
        except SystemExit:
            raise
        except Exception as e:
            say("  失败：%s: %s" % (type(e).__name__, e))
            return 1
    say("=" * 60)
    say("完成。核对清单见 assets/SOURCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
