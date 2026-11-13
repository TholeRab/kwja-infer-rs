# -*- coding: utf-8 -*-
"""用 onnxsim 化简 KWJA 的两个 ONNX 模型（常量折叠＋死代码消除）。

为什么有这个脚本（）：tract 直接加载 torch 导出的 DeBERTa 时，
在常量子图上连环卡壳（i64 Sign、Clip 的 dtype 统一、StridedSlice 符号维等）。
onnxsim 把可折叠的常量计算在模型层面折掉、清理 dtype 与死节点，
让 tract 的形状推理/常量求值不再碰到这些边角。

用法：python simplify_models.py <model.onnx> [更多.onnx ...]
原地改写；dynamic 维保持 dynamic。
"""
import sys
import onnx
from onnxsim import simplify


def convert(path: str) -> None:
    model = onnx.load(path)
    simplified, check = simplify(model)
    assert check, f"{path}: onnxsim 校验失败"
    onnx.save(simplified, path)
    print(f"{path}: onnxsim 完成，节点数 {len(model.graph.node)} -> {len(simplified.graph.node)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
