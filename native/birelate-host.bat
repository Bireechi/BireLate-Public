@echo off
REM Firefox launches a native messaging host as an executable, and a .ps1 is not
REM one. This wrapper is the executable; it does nothing but hand stdio to
REM PowerShell.
REM
REM -NoProfile matters more than it looks: a profile that prints anything at all
REM -- a banner, a version notice, an oh-my-posh prompt -- writes it to stdout,
REM which is the length-prefixed message stream. The first byte of noise
REM desynchronises Firefox's reader and the host looks silently broken.
REM Started from a PowerShell 7 terminal, PSModulePath points at PS 7's modules
REM and Windows PowerShell then loses cmdlets such as Get-FileHash. Cleared, it
REM rebuilds its own default.
set "PSModulePath="
powershell.exe -NoProfile -NoLogo -NonInteractive -ExecutionPolicy Bypass -File "%~dp0birelate-host.ps1"
