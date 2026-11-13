@echo off
rem 打发行包：先构建，再把 exe 与 assets 归集到 dist\kwja-infer-<提交号>\。
rem 用法：tools\package-release.cmd（无需管理员）。
setlocal
cd /d "%~dp0.."
call tools\build-release.cmd -p kwja-infer
if errorlevel 1 exit /b 1
for /f %%i in ('git rev-parse --short HEAD') do set "SHA=%%i"
set "OUT=%CD%\target\release\kwja-infer-%SHA%"
echo 出包目录：%OUT%
if not exist "%OUT%" mkdir "%OUT%"
copy /y target\release\kwja-infer.exe "%OUT%\" >nul
robocopy assets "%OUT%\assets" /E /MT:16 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1
copy /y tools\发行说明-template.txt "%OUT%\说明.txt" >nul
dir /b "%OUT%"
echo 完成：%OUT%
