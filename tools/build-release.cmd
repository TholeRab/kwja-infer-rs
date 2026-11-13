@echo off
rem 仓库内 release 构建入口（理由见 build-release.sh 头注）。
setlocal
set "HERE=%~dp0.."
set "CARGO_TARGET_DIR="
if not exist "%HERE%\link-tmp" mkdir "%HERE%\link-tmp"
set "TMP=%HERE%\link-tmp"
set "TEMP=%HERE%\link-tmp"
cargo build --release %*
