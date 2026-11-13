# -*- coding: utf-8 -*-
"""把 word_encoder7.onnx 里唯一的 i64 Sign 节点改写成 f32 Sign（前后各补一次 Cast）。

为什么有这个脚本（）：tract-onnx 0.21.12 的 Sign 遗留求值不支持 I64
（「Sign does not support I64」），而 DeBERTa 的相对位置分桶在导出时留下了
对常量做 i64 Sign 的节点。Sign 的输入是小的整数（位置量），经 f32 中转
逐位精确：i64→f32 精确、sign 结果 -1/0/1 精确、f32→i64 精确。
ORT 对两种写法结果相同，模型语义不变。

用法：python patch_word_sign_i64.py <word_encoder7.onnx>
"""
import sys
import onnx
from onnx import TensorProto, helper


def convert(path: str) -> None:
    model = onnx.load(path)
    graph = model.graph
    sign = next(n for n in graph.node if n.op_type == "Sign")
    src = sign.input[0]
    dst = sign.output[0]
    idx = list(graph.node).index(sign)
    graph.node.remove(sign)
    casts = [
        helper.make_node(
            "Cast", [src], [src + "_f32"],
            name="sign_wrap_cast_in", to=TensorProto.FLOAT,
        ),
        helper.make_node("Sign", [src + "_f32"], [dst + "_f32"], name="sign_wrap_sign"),
        helper.make_node(
            "Cast", [dst + "_f32"], [dst],
            name="sign_wrap_cast_out", to=TensorProto.INT64,
        ),
    ]
    for offset, node in enumerate(casts):
        graph.node.insert(idx + offset, node)
    onnx.save(model, path)
    print(f"{path}: Sign(i64) 已改写为 Cast→Sign(f32)→Cast")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
