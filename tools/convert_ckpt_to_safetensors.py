# -*- coding: utf-8 -*-
"""把 KWJA 的 Lightning checkpoint 转成 candle 可直接加载的 safetensors。

为什么有这个脚本（）：路线 A（手写 BLAS 编码器，基于 candle 的
debertav2 参考实现）需要 HF 命名的权重；KWJA 的权重存在 Lightning ckpt 里，
state_dict 键带 "encoder." 前缀（Lightning 模块属性名）。本脚本：
1. 剥掉 "encoder." 前缀（candle debertav2 的 VarBuilder 命名与 HF 一致，
   query_proj 等键名与 ckpt 完全同名，见 candle-transformers 0.9.2
   debertav2.rs 的 load）；
2. char 模型的三个标签头（sent_segmentation_tagger 等）不是 encoder 的子模块，
   键名原样保留；
3. 落成 safetensors，供 candle VarBuilder::from_tensors 加载。

用法：python convert_ckpt_to_safetensors.py
"""
from pathlib import Path

import torch
from safetensors.torch import save_file

CACHE = Path("外部素材目录/kwja/v2.4")
OUT = Path("assets/models")

JOBS = [
    (CACHE / "word_deberta-v2-base.ckpt", OUT / "word_deberta-v2-base.safetensors"),
    (CACHE / "char_deberta-v2-base-wwm.ckpt", OUT / "char_deberta-v2-base-wwm.safetensors"),
]


def main() -> None:
    for src, dst in JOBS:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        converted = {}
        for key, value in state.items():
            name = key[len("encoder."):] if key.startswith("encoder.") else key
            converted[name] = value.contiguous()
        save_file(converted, str(dst))
        print(f"{dst.name}: {len(converted)} tensors")
        heads = [k for k in converted if "tagger" in k]
        print("  tagger keys:", heads[:6])


if __name__ == "__main__":
    main()
