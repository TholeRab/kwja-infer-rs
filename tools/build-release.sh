#!/usr/bin/env bash
# 仓库内 release 构建入口。直接 cargo build --release 在本机有两个坑：
# 1) 用户级环境变量 CARGO_TARGET_DIR=C:\cargo-target 把产物指到 C 盘；
# 2) 目标目录在 V 盘时，onig_sys 的 MSVC lib.exe 在系统 %TEMP% 建临时文件
#    会报 LNK1104（疑似杀软锁文件），退到家目录同样失败。
# 本脚本清掉前者、把 TMP/TEMP 指到仓库内常驻的 link-tmp/（实测可过），
# 产物落在仓库 target/release/。
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
unset CARGO_TARGET_DIR
mkdir -p "$root/link-tmp"
# MSYS 的 pwd 给出 /v/... 形路径，原生 link.exe 会把它按当前盘错拆成 C:\v\...；
# 必须先转成 Windows 绝对路径再交给 cargo/link.exe。
tmpdir="$(cygpath -w "$root/link-tmp")"
export TMP="$tmpdir" TEMP="$tmpdir"
exec cargo build --release "$@"
