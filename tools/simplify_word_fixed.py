# -*- coding: utf-8 -*-
"""按固定输入形状重跑 onnxsim，把随 seq 变化的常量子图折成字面常量。

为什么有这个脚本（）：动态形状下，onnxsim 折不掉「位置分桶」那类
依赖 seq 的常量计算；tract 的形状推理把它们推成 TDim 类型的张量，遇到
Clip 这类要求 dtype 一致的规则就报「Impossible to unify TDim with I64」。
word 编码器在本分支恒为 (1, 256)（max_seq_length），按固定形状简化后
整图静态化，tract 直接吃静态图。

用法：python simplify_word_fixed.py <word_encoder7.onnx>
"""
import sys
import onnx
from onnxsim import simplify


def convert(path: str) -> None:
    model = onnx.load(path)
    simplified, check = simplify(
        model,
        overwrite_input_shapes={
            "input_ids": [1, 256],
            "attention_mask": [1, 256],
            "special_token_indices": [1, 7],
        },
    )
    assert check, f"{path}: onnxsim 校验失败"
    onnx.save(simplified, path)
    print(f"{path}: 定形简化完成，节点数 {len(model.graph.node)} -> {len(simplified.graph.node)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
