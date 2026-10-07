# Build environment for Koharu on Windows (MSVC).
# Puts cargo, CMake, Ninja and LLVM (libclang + clang-cl) on PATH so the
# -sys build scripts can find them.
#
# Every argument is passed to cargo verbatim, so the profile is chosen the
# ordinary cargo way and this script needs no switch of its own:
#     build-koharu.ps1 check -p birelate-server --all-targets
#     build-koharu.ps1 test  -p birelate-server
#     build-koharu.ps1 build --release -p birelate-server
# A first release build is a full from-cold rebuild of the whole graph, so
# expect it to take a long time. Nothing else needs to change for it: cargo's
# built-in release profile is already opt-level 3 for every package, and
# koharu-torch-sys derives its DLL destination from OUT_DIR, so
# target\release\koharu-torch.dll appears on its own.
#
# NOTE: do not set ErrorActionPreference to Stop here. Cargo writes its normal
# progress output to stderr, and PowerShell 5.1 wraps native stderr lines as
# NativeCommandError records, which would abort the build on the first line.
# For the same reason, do not pipe this script into anything: the wrapping
# produces a bogus exit 255 on a build that actually succeeded. Run it bare and
# read the "--- cargo exit: N" line.
$ErrorActionPreference = "Continue"

# The repo MUST live on a short path. MSVC/MSBuild still enforce the 260-char
# MAX_PATH limit, and cargo appends a deep target\<profile>\build\<crate>-<hash>\out\...
# prefix to every -sys crate's CMake output. Building under a long path fails
# with MSB3191 "path too long".
. "$PSScriptRoot\local-paths.ps1"
$repo = Get-BireLatePath koharu_root
$bt   = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"

function Add-ToPath($dir) {
    if ($dir -and (Test-Path $dir)) { $env:PATH = "$dir;$env:PATH"; return $true }
    return $false
}

# cargo
Add-ToPath "$env:USERPROFILE\.cargo\bin" | Out-Null

# LLVM: libclang for bindgen, clang-cl for the libtch shim
$llvm = "C:\Program Files\LLVM\bin"
if (Add-ToPath $llvm) { $env:LIBCLANG_PATH = $llvm }

# Ninja: required by koharu-torch-sys's CMake invocation on Windows.
# winget drops it in a Packages dir and only edits the persistent PATH, so an
# already-running shell won't see it.
Add-ToPath "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe" | Out-Null
Add-ToPath "$env:LOCALAPPDATA\Microsoft\WinGet\Links" | Out-Null

# CMake: prefer standalone, fall back to the VS Build Tools copy
foreach ($c in @("C:\Program Files\CMake\bin",
                 "$bt\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin")) {
    if (Test-Path "$c\cmake.exe") { Add-ToPath $c | Out-Null; break }
}

foreach ($t in 'cargo','cmake','ninja','clang-cl') {
    $g = Get-Command $t -ErrorAction SilentlyContinue
    if ($g) { Write-Host ("{0,-9}: {1}" -f $t, $g.Source) }
    else    { Write-Host ("{0,-9}: NOT FOUND" -f $t) }
}
Write-Host ("{0,-9}: {1}" -f "LIBCLANG", $env:LIBCLANG_PATH)
Write-Host "---"

# birelate-server's source of truth is the server\ folder beside scripts\. The workspace has
# members = ["crates/*"], so a junction under crates\ makes it a member without
# copying anything -- and the build still happens under the short path.
$link = Join-Path $repo "crates\birelate-server"
$src  = Join-Path $BireLatePathRoot "server"
if (Test-Path $src) {
    if (-not (Test-Path $link)) {
        # A junction whose target has moved still occupies the name while
        # Test-Path reports it missing, so mklink would fail with "file already
        # exists" and cargo would then fail somewhere much less obvious.
        $stale = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
        if ($stale) {
            Write-Host "junction : removing a dangling $link"
            $stale.Delete()
        }
        Write-Host "junction : recreating $link -> $src"
        cmd /c mklink /J "$link" "$src" | Out-Null
    }
} else {
    Write-Host "junction : $src is missing; birelate-server will not resolve"
}

Set-Location $repo
cargo @args
$code = $LASTEXITCODE
Write-Host "--- cargo exit: $code"

# Name the directory the binaries landed in, so this and serve.ps1's "looked in"
# message can be compared without anyone having to remember cargo's mapping.
if ($code -eq 0) {
    $out = "debug"
    if ($args -contains "--release") { $out = "release" }
    for ($i = 0; $i -lt $args.Count - 1; $i++) {
        # --profile wins over --release: it is the more specific flag, and cargo
        # rejects the two together anyway.
        if ($args[$i] -eq "--profile") { $out = $args[$i + 1] }
    }
    if ($out -eq "dev") { $out = "debug" }   # the dev profile builds into target\debug
    Write-Host "--- output    : $repo\target\$out"
}

exit $code
