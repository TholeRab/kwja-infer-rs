# -*- coding: utf-8 -*-
# 流式逐行对比两份 KNP：内存 O(1)，只收集差异行。整书对拍一律用本脚本，不要用 diff。
# 用法：
#   python compare_knp.py <A.knp> <B.knp>                          纯报告
#   python compare_knp.py <A.knp> <B.knp> --emit-registry <out>    把当前差异写成豁免登记
#   python compare_knp.py <A.knp> <B.knp> --expected <json>        按豁免登记验收（闸）
# 验收口径：实际差异集合必须与登记完全一致（行号与两侧内容逐字相等），多一行少一行都算失败。
import argparse, io, json, sys


def lines(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            yield ln.rstrip("\r\n")


def compare(a_path, b_path):
    na = nb = 0
    diffs = []
    ga, gb = lines(a_path), lines(b_path)
    while True:
        xa = next(ga, None)
        xb = next(gb, None)
        if xa is None and xb is None:
            break
        if xa is not None:
            na += 1
        if xb is not None:
            nb += 1
        if xa != xb:
            diffs.append({"line": na, "rust": xa if xa is not None else None,
                          "reference": xb if xb is not None else None})
    return na, nb, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--emit-registry", metavar="OUT")
    ap.add_argument("--expected", metavar="JSON")
    ap.add_argument("--note", metavar="TEXT", help="写登记时附带的说明文字")
    args = ap.parse_args()

    na, nb, diffs = compare(args.a, args.b)
    print("A 行数=%d  B 行数=%d  差异行数=%d" % (na, nb, len(diffs)))
    for d in diffs:
        print("--- 第%d行" % d["line"])
        print("  A: %s" % (d["rust"] if d["rust"] is not None else "<缺行>"))
        print("  B: %s" % (d["reference"] if d["reference"] is not None else "<缺行>"))

    if args.emit_registry:
        with io.open(args.emit_registry, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"note": args.note or "", "diffs": diffs}, f,
                      ensure_ascii=False, indent=2)
            f.write("\n")
        print("已写豁免登记 -> %s（%d 条）" % (args.emit_registry, len(diffs)))
        return 0

    if args.expected:
        with io.open(args.expected, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        expected = loaded["diffs"] if isinstance(loaded, dict) else loaded
        got = {(d["line"], d["rust"], d["reference"]) for d in diffs}
        want = {(d["line"], d["rust"], d["reference"]) for d in expected}
        bad = []
        for d in sorted(got - want):
            bad.append("未登记的差异：第%d行" % d[0])
        for d in sorted(want - got):
            bad.append("登记已失效（实际无此差异或内容已变）：第%d行" % d[0])
        if bad:
            print("对拍闸：不通过")
            for b in bad:
                print("  " + b)
            return 1
        print("对拍闸：通过（差异恰为登记的 %d 条已知数值翻转）" % len(expected))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
