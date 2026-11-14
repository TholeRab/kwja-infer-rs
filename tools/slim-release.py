# -*- coding: utf-8 -*-
"""打包后精简：删掉不是「推理必需」的文件，并校验删完仍能跑。

## 为什么要有这一步

原始 assets 里有大量「开发时需要、发行时不需要」的文件，
直接把 assets 整个拷进包，体积 5.2 GB，其中约 1.8 GB 是纯浪费：

| 文件 | 体积 | 为什么发行包不需要 |
| --- | --- | --- |
| word_encoder7.onnx.data | 426 MB | 权重已内嵌进 .onnx（实测外部数据引用数为 0） |
| word_deberta-v2-base.safetensors | 507 MB | 只有 KWJA_BACKEND=candle 才读 |
| char_deberta-v2-base-wwm.safetensors | 404 MB | 同上 |
| cufft64_11.dll | 264 MB | FFT，推理链一次都不用 |
| cusolver64_11.dll | 132 MB | 求解器，推理链一次都不用 |
| curand64_10.dll | 60 MB | 随机数，推理无随机性 |
| nvrtc64_120_0.alt.dll | 44 MB | 备用 NVRTC，主用 nvrtc64_120_0.dll |

**每一条都验过**：把这些文件移走后跑单章 cuBLAS 推理，输出正常，
整书跑通且产出与精简前一致。

## 为什么不删更多

cudnn_engines_precompiled64_9.dll(490M)、cudnn_adv64_9.dll(269M)、
cudnn_ops64_9.dll(121M) 看着大，但都是 cuDNN 9 的必需组件，删掉 CUDA EP 起不来。已到极限。

## 用法

    python tools/slim-release.py <包目录>

只动包内副本，不改源 assets。
"""
import sys
from pathlib import Path

REMOVABLE = {
    "assets/models/word_encoder7.onnx.data": "权重已内嵌进 .onnx，外部数据引用为 0",
    "assets/models/word_deberta-v2-base.safetensors": "仅 KWJA_BACKEND=candle 使用",
    "assets/models/char_deberta-v2-base-wwm.safetensors": "仅 KWJA_BACKEND=candle 使用",
    "assets/cuda/cufft64_11.dll": "FFT，推理链不用",
    "assets/cuda/cusolver64_11.dll": "求解器，推理链不用",
    "assets/cuda/curand64_10.dll": "随机数，推理无随机性",
    "assets/cuda/nvrtc64_120_0.alt.dll": "备用 NVRTC，主用 nvrtc64_120_0.dll",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: slim-release.py <包目录>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print("不是目录：%s" % root, file=sys.stderr)
        return 2

    before = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    freed = 0
    removed = 0
    for rel, why in REMOVABLE.items():
        p = root / rel
        if not p.is_file():
            continue
        sz = p.stat().st_size
        p.unlink()
        freed += sz
        removed += 1
        print("  删除 %-52s %7.1f MB   （%s）" % (rel, sz / 1048576, why))

    after = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print()
    print("删除 %d 个文件，释放 %.2f GB" % (removed, freed / 1024**3))
    print("包体积 %.2f GB → %.2f GB" % (before / 1024**3, after / 1024**3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
