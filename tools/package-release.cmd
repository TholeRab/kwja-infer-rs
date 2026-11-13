@echo off
rem Package a release build: build -> collect exe + assets -> ../release/<name>
rem
rem Why ../release/ and not target/:
rem   target/ is cargo build dir (gitignored). cargo clean wipes it, and a
rem   failed build can leave it dirty. A release package must not live
rem   somewhere that gets deleted.
rem
rem Why the directory name carries a date:
rem   A bare commit hash does not tell the recipient which is newer.
rem   Date first sorts naturally; hash second traces back to code.
rem
rem Usage: tools\package-release.cmd [--skip-build]
rem
rem NOTE: deliberately ASCII-only, CRLF only. cmd.exe parses .cmd in the
rem system ANSI codepage (cp936 here); UTF-8 Chinese comments break it.
rem Also: "::" comments must not contain ";", and backslash escapes in
rem generated text corrupt paths silently (both observed here once).
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
set "OUTROOT=%ROOT%\..\release"

if /i not "%~1"=="--skip-build" (
    call tools\build-release.cmd -p kwja-infer
    if errorlevel 1 exit /b 1
)

set "EXE=%ROOT%\target\release\kwja-infer.exe"
if not exist "%EXE%" (
    echo ERROR: %EXE% not found, build first
    exit /b 1
)

for /f %%i in ('git rev-parse --short HEAD') do set "SHA=%%i"
for /f %%i in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%i"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "DATE=%%i"

set "OUT=%OUTROOT%\kwja-infer-%DATE%-%SHA%"

echo branch: %BRANCH%  commit: %SHA%
echo output: %OUT%

if exist "%OUT%" (
    echo ERROR: target directory already exists: %OUT%
    echo        remove it or use a different commit
    exit /b 1
)

mkdir "%OUT%"
if errorlevel 1 exit /b 1
copy /y "%EXE%" "%OUT%\" >nul
if errorlevel 1 exit /b 1

rem assets is about 5 GB; /MIR so no stale leftovers
robocopy "%ROOT%\assets" "%OUT%\assets" /MIR /MT:16 /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 (
    echo ERROR: assets copy failed, robocopy exit code %errorlevel%
    exit /b 1
)

rem Slim: drop dev-only files (about 1.8 GB). List + verification in
rem tools/slim-release.py.
python "%ROOT%\tools\slim-release.py" "%OUT%"
if errorlevel 1 exit /b 1

rem Release notes via Python: PowerShell decodes with cp936 and mangles
rem UTF-8 Chinese (observed once: whole file turned to garbage).
python "%ROOT%\tools\write-release-notes.py" "%ROOT%\tools\release-notes-template.txt" "%OUT%\README.txt" "%SHA%" "%BRANCH%" "%DATE%"
if errorlevel 1 exit /b 1

echo.
echo contents:
dir /b "%OUT%"
echo.
echo done: %OUT%
