# -*- coding: utf-8 -*-
"""把发行说明模板写成具体版本：替换占位符并写入目标文件。

为什么单独一个脚本、而不在 .cmd 里用 PowerShell 做替换：
PowerShell 的 Get-Content 默认按系统 ANSI 解码（本机 cp936），
读 UTF-8 的中文模板会读坏，再写出去就是乱码（实测过一次，整份说明报废）。
这里显式 encoding="utf-8"，行为与平台无关。

用法（由 package-release.cmd 调用）：
    python 写发行说明.py <模板> <输出> <提交号> <分支> <日期>
"""
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 6:
        print("用法: 写发行说明.py <模板> <输出> <提交号> <分支> <日期>", file=sys.stderr)
        return 2
    tpl, out, sha, branch, date = sys.argv[1:]
    text = Path(tpl).read_text(encoding="utf-8")
    text = text.replace("版本号见目录名", f"提交 {sha} / 分支 {branch} / {date}")
    Path(out).write_text(text, encoding="utf-8", newline="\n")
    print(f"说明已写入：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
