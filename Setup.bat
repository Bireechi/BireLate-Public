@echo off
REM One-time setup: checks this PC, installs the CUDA runtime and, if you accept
REM its licence, HunyuanOCR. Safe to run again. Options: -AcceptHunyuanLicense,
REM -SkipHunyuan, -NoNativeHost, -DryRun.
setlocal
REM Run from inside a zip viewer, only this file is extracted.
if not exist "%~dp0scripts\setup.ps1" (
    echo scripts\setup.ps1 is missing: extract the whole zip to a folder first,
    echo then run Setup.bat from that folder.
    pause
    exit /b 1
)
REM Started from a PowerShell 7 terminal, PSModulePath points at PS 7's modules
REM and Windows PowerShell then loses cmdlets such as Get-FileHash. Cleared, it
REM rebuilds its own default.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set "RC=%ERRORLEVEL%"
REM Always pause: the summary at the end says what to do next.
pause
exit /b %RC%
