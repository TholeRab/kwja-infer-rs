# -*- coding: utf-8 -*-
"""把 ONNX Constant 节点的 value_ints/value_floats 属性改写成标准 value 张量属性。

为什么有这个脚本（）：tract-onnx 0.21.12 的 Constant 解析器只认
value / value_int / value_float 三种属性（tract-onnx src/ops/mod.rs konst），
不认 value_ints / value_floats；而 KWJA 的 char_dynamic.onnx 里有 722 个
value_ints 常量，tract 一加载就报
「Could not extract value out of Constant node」。
ORT 对 value_ints 的语义是「一维 INT64 张量」（ONNX 规范同此），
所以改写成显式的一维 value 张量与 ORT 行为逐位一致，不改变模型语义。

用法：python normalize_onnx_constants.py <model.onnx> [更多.onnx ...]
原地改写；只动 Constant 节点的属性，其余结构一律不碰。
"""
import sys
import onnx
from onnx import TensorProto, helper


def convert(path: str) -> None:
    model = onnx.load(path)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in list(node.attribute):
            if attr.name == "value_ints":
                dims = [len(attr.ints)]
                tensor = helper.make_tensor(
                    node.output[0], TensorProto.INT64, dims, list(attr.ints)
                )
            elif attr.name == "value_floats":
                dims = [len(attr.floats)]
                tensor = helper.make_tensor(
                    node.output[0], TensorProto.FLOAT, dims, list(attr.floats)
                )
            else:
                continue
            node.attribute.remove(attr)
            node.attribute.append(helper.make_attribute("value", tensor))
            changed += 1
    onnx.save(model, path)
    print(f"{path}: 改写 {changed} 个 Constant 节点")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
