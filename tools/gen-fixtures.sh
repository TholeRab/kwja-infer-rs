#!/usr/bin/env bash
# 生成测试基线（fixtures）。这些文件不进 git：共约 36 MB，且完全可重建。
#
# 为什么不进 git：
#   - word_knp_all.txt 21 MB，GitHub 单文件上限 100 MB，但每次改输出都要重提交一份，
#     仓库会随历史线性膨胀；
#   - 它们全都能从 corpus/ 重新算出来，corpus/ 只有 856 KB，那个才该进 git。
#
# 依赖：参考实现仓 仓（提供导出脚本）与它的 Python venv。
#
# 用法：
#   tools/gen-fixtures.sh              # 全部（约 10 分钟，要 GPU）
#   tools/gen-fixtures.sh --char-only  # 只生成 char_stage_*
#
set -e

RS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORJ="${ORJ_ROOT:-$RS_ROOT/../参考实现仓}"
PY="${KWJA_PYTHON:-$RS_ROOT/../Python 环境/Scripts/python.exe}"

if [ ! -d "$ORJ" ]; then
    echo "错误：找不到 参考实现仓：$ORJ" >&2
    echo "      设 ORJ_ROOT 指向它" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "错误：找不到 Python：$PY" >&2
    echo "      设 KWJA_PYTHON 指向装了 kwja + torch(cu126) 的解释器" >&2
    exit 1
fi

echo "仓库：$RS_ROOT"
echo "orj ：$ORJ"
echo "py  ：$PY"
echo ""

# ---- 阶段 5：corpus → char_stage_*.txt ----
echo "=== 阶段5：char_stage_*（读 corpus/，产出 char 阶段 Juman++ 文本）==="
cd "$ORJ"
"$PY" -X utf8 阶段一/导出Rust侧阶段5基准.py --chapters 1,3,6,10,59

if [ "${1:-}" = "--char-only" ]; then
    echo ""
    echo "只生成 char 基线，停在这里。"
    exit 0
fi

# ---- 阶段 7：char_stage_* → word_knp_*.txt ----
echo ""
echo "=== 阶段7：word_knp_*（读 char_stage_*，产出 KNP 文本）==="
"$PY" -X utf8 阶段一/导出Rust侧阶段7权重与基准.py --baselines-only --tiers

# ---- 阶段 6：分窗结构 ----
echo ""
echo "=== 阶段6：word_windows_*（分窗结构基线）==="
"$PY" -X utf8 阶段一/导出Rust侧阶段6基准.py

echo ""
echo "=== 完成，产物 ==="
cd "$RS_ROOT"
ls -la crates/pipeline/tests/fixtures/*.txt | awk '{printf "%8.1f MB  %s\n", $5/1048576, $9}'
