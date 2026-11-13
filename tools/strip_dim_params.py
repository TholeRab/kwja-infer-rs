# -*- coding: utf-8 -*-
"""去掉 ONNX 图输入/输出上的 dim_param 名（batch/seq 等），只留未知维。

为什么有这个脚本（）：tract 的形状推理会把 dim_param 提升为具名符号，
与代码里 set_input_fact 传入的具体形状做统一时冲突
（「Impossible to unify Sym(batch) with Val(1)」）。
dim_param 只是元数据；ORT 与推理结果都不依赖它。去掉后 tract 用匿名维，
具体事实可以正常代入。配合 tools/simplify_models.py 使用。

用法：python strip_dim_params.py <model.onnx> [更多.onnx ...]
"""
import sys
import onnx


def convert(path: str) -> None:
    model = onnx.load(path)
    stripped = 0
    for tensor_type in (
        [v.type for v in model.graph.value_info]
        + [i.type for i in model.graph.input]
        + [o.type for o in model.graph.output]
    ):
        for dim in tensor_type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.ClearField("dim_param")
                stripped += 1
    onnx.save(model, path)
    print(f"{path}: 去掉 {stripped} 个 dim_param")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
