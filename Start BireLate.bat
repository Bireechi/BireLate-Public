@echo off
REM Starts the BireLate server. Close this window to stop it.
REM Options are passed to scripts\serve.ps1, e.g. -DryRun to see what would run.
setlocal
REM serve.ps1 leaves the pause on failure to this file, so it is asked once.
set "BIRELATE_BAT=1"
REM Started from a PowerShell 7 terminal, PSModulePath points at PS 7's modules
REM and Windows PowerShell then loses cmdlets such as Get-FileHash. Cleared, it
REM rebuilds its own default.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\serve.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
