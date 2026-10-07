<#
.SYNOPSIS
    One-time setup for BireLate. Safe to run again at any time.

.DESCRIPTION
    Checks the machine, then installs what the BireLate zip does not ship:
      - the CUDA runtime DLL the server's GPU probe looks for at startup (always)
      - HunyuanOCR, the default OCR engine, with llama.cpp and a small Python to
        serve it -- only if you accept its licence; otherwise BireLate uses
        PaddleOCR-VL, which Koharu downloads by itself on first use
      - the native messaging host that lets the extension's Start button launch
        the server (one HKCU registry key)

    Everything lands under local\ in this folder. Each file is checked against a
    pinned size and SHA256: a file already in place is not downloaded again, and
    an interrupted download resumes on the next run. Koharu's own runtimes and
    models are NOT downloaded here; the server fetches them on first use.

.PARAMETER AcceptHunyuanLicense
    Accept the Tencent Hunyuan Community License without being asked.

.PARAMETER SkipHunyuan
    Do not install HunyuanOCR, the main OCR engine; BireLate will use
    PaddleOCR-VL, which lowers quality, especially for manhua and manhwa.

.PARAMETER NoNativeHost
    Do not register the native messaging host.

.PARAMETER DryRun
    Print the plan; download nothing and change nothing.
#>

param(
    [switch]$AcceptHunyuanLicense,
    [switch]$SkipHunyuan,
    [switch]$NoNativeHost,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\local-paths.ps1"
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem

$LicenseUrl = "https://huggingface.co/tencent/HunyuanOCR/blob/449e7d471a8a1ef5bd5d652e4881183d7252cbc7/LICENSE"
$Curl       = Join-Path $env:SystemRoot "System32\curl.exe"
$Downloads  = Get-BireLatePath downloads
$State      = Get-BireLatePath state
$Hunyuan    = Get-HunyuanFiles
$CudartPath = Join-Path (Get-BireLatePath cuda_bootstrap) "cudart64_13.dll"

# Every download, pinned. Sizes and hashes are the publishers' own: PyPI's JSON
# API, the GitHub release API, the Hugging Face tree API, and the SPDX SBOM
# python.org publishes beside the embeddable zip.
$Pins = @{
    cudart = @{ Name = "nvidia_cuda_runtime-13.0.48-py3-none-win_amd64.whl"; Size = 2926656
        Sha256 = "03e581c7584b13e42ce175c774f46e1219e9c574f27fe88c2ccc75dd3f926ed7"
        Url = "https://files.pythonhosted.org/packages/b2/dc/43b84c49f938817626370ce2c7dbb9d6f9a3a0f9d9764e5902501e106e82/nvidia_cuda_runtime-13.0.48-py3-none-win_amd64.whl" }
    # Not downloaded: extracted from the wheel above. The same file Koharu installs.
    cudartDll = @{ Name = "cudart64_13.dll"; Size = 470640; Entry = "nvidia/cu13/bin/x86_64/cudart64_13.dll"
        Sha256 = "58c509047f9dd1879cf82c6d5375a95c02c7bd295ae6c8362c745c936e2afc31" }
    llama = @{ Name = "llama-b10502-bin-win-cuda-13.3-x64.zip"; Size = 146813754
        Sha256 = "657ad104b7c2f3aaf9abac91b48ffb72a2556cb8a6a38d395eaaf64bc1f1f719"
        Url = "https://github.com/ggml-org/llama.cpp/releases/download/b10502/llama-b10502-bin-win-cuda-13.3-x64.zip" }
    llamaCudart = @{ Name = "cudart-llama-bin-win-cuda-13.3-x64.zip"; Size = 390970417
        Sha256 = "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e"
        Url = "https://github.com/ggml-org/llama.cpp/releases/download/b10502/cudart-llama-bin-win-cuda-13.3-x64.zip" }
    model = @{ Name = "HunyuanOCR.BF16.gguf"; Size = 1083218528
        Sha256 = "489dc42338cac27b1d93b7f503b5df65d8e829dd33b43bad94227d929d8a4541"
        Url = "https://huggingface.co/prithivMLmods/HunyuanOCR-1.5-GGUF-Updated/resolve/9ddd3b47beb0de305ecd89a717748bac080d7aee/HunyuanOCR.BF16.gguf" }
    mmproj = @{ Name = "HunyuanOCR.mmproj-bf16.gguf"; Size = 997235744
        Sha256 = "2c9c459f68a9a3c221b1a8088d9c91ac1007ef22aa89045108d14d949f9a3994"
        Url = "https://huggingface.co/prithivMLmods/HunyuanOCR-1.5-GGUF-Updated/resolve/9ddd3b47beb0de305ecd89a717748bac080d7aee/HunyuanOCR.mmproj-bf16.gguf" }
    template = @{ Name = "chat_template.jinja"; Size = 994
        Sha256 = "be3371395b9e67a8f981d86543eb5a93d132a1dc3f54058a2d75b4ed1efc73fe"
        Url = "https://huggingface.co/tencent/HunyuanOCR/resolve/449e7d471a8a1ef5bd5d652e4881183d7252cbc7/chat_template.jinja" }
    # Tencent's licence text, kept beside the model it governs.
    license = @{ Name = "HunyuanOCR-LICENSE"; Size = 16277
        Sha256 = "745adaa59575d2a98b64fd6d3452537b477a6b6edc126f742fc055313cc3d3e0"
        Url = "https://huggingface.co/tencent/HunyuanOCR/resolve/449e7d471a8a1ef5bd5d652e4881183d7252cbc7/LICENSE" }
    python = @{ Name = "python-3.12.10-embed-amd64.zip"; Size = 11133606
        Sha256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
        Url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" }
    pillow = @{ Name = "pillow-12.3.0-cp312-cp312-win_amd64.whl"; Size = 7227137
        Sha256 = "a2b55dd6b2a4c4b7d87ffa56bdb33fdc5fdb9a462173861a7bc097f17d91cb09"
        Url = "https://files.pythonhosted.org/packages/45/89/da2f7971a317f83d807fdd4065c0af40208e59e692cc43d315a71a0e96d1/pillow-12.3.0-cp312-cp312-win_amd64.whl" }
}

# ---- checks -----------------------------------------------------------------

# Warns about everything it can; returns $false only for what makes setup or the
# server impossible.
function Test-Prerequisites {
    $ErrorActionPreference = "Continue"   # native stderr must not abort a check
    $ok = $true

    if ([Environment]::Is64BitOperatingSystem) {
        Write-Host "windows : 64-bit"
    } else {
        Write-Host "windows : 32-bit Windows is not supported; BireLate needs 64-bit Windows 10 or 11"
        $ok = $false
    }

    # 14.40 is the floor: koharu-torch.dll is built with the VS 2022 17.10+ C++
    # library, whose std::mutex crashes (0xC0000005) on an older msvcp140.dll.
    # The two keys can disagree; the newer one is what is installed.
    $vc = $null
    $vcText = ""
    foreach ($key in @("HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                       "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64")) {
        $v = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
        if (-not $v -or $v.Installed -ne 1) { continue }
        $ver = [version]"0.0"
        try {
            $ver = if ($null -ne $v.Major) { [version]("{0}.{1}" -f $v.Major, $v.Minor) }
                   else { [version]("$($v.Version)".TrimStart('v')) }
        } catch {}
        if (-not $vc -or $ver -gt $vc) { $vc = $ver; $vcText = "$($v.Version)" }
    }
    if (-not $vc) {
        Write-Host "vc++    : the Visual C++ 2015-2022 x64 runtime is NOT installed. Install it, then run Setup.bat again:"
        Write-Host "          https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $ok = $false
    } elseif ($vc -lt [version]"14.40") {
        Write-Host "vc++    : runtime $vcText is too old; BireLate needs 14.40 or newer. Install the latest"
        Write-Host "          even if an older one is present, then run Setup.bat again:"
        Write-Host "          https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $ok = $false
    } else {
        Write-Host ("vc++    : runtime {0}" -f $vcText)
    }

    $smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Path
    if (-not $smi) { $smi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe" }
    $gpus = @()
    if (Test-Path -LiteralPath $smi) {
        $gpus = @(& $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>$null |
            Where-Object { $_ -match ',' })
    }
    if ($gpus.Count -eq 0) {
        Write-Host "gpu     : WARNING no NVIDIA GPU or driver found (nvidia-smi). BireLate needs an NVIDIA GPU."
    }
    foreach ($line in $gpus) {
        $name, $driver, $mib = $line -split ',\s*'
        Write-Host ("gpu     : {0}, driver {1}, {2} MiB" -f $name, $driver, $mib)
    }
    if ($gpus.Count -gt 0) {
        # CUDA uses device 0 unless told otherwise, so that is the one checked.
        $name, $driver, $mib = $gpus[0] -split ',\s*'
        if ((($driver -split '\.')[0] -as [int]) -lt 580) {
            Write-Host "          WARNING driver $driver is too old: BireLate needs R580 or newer (CUDA 13)."
            Write-Host "          Update it from https://www.nvidia.com/drivers"
        }
        # 23000, not 24 * 1024: a 24 GB card such as the RTX 4090 reports ~24564 MiB.
        if (($mib -as [int]) -lt 23000) {
            Write-Host "          WARNING less than 24 GB of VRAM. The default translator (gemma4-26b-a4b-it)"
            Write-Host "          needs about 21 GB free. Pick a smaller one with: Start BireLate.bat -Llm <model>"
            Write-Host "          (see README.md); -ColdReserveBytes tunes the VRAM pre-flight."
        }
    }

    if (Test-Path -LiteralPath $Curl) {
        Write-Host "curl    : $Curl"
    } else {
        Write-Host "curl    : $Curl not found. It ships with Windows 10 1803 and later; setup cannot download without it."
        $ok = $false
    }

    try {
        $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($BireLatePathRoot))
        $freeGb = $drive.AvailableFreeSpace / 1GB
        Write-Host ("disk    : {0:N1} GB free on {1}" -f $freeGb, $drive.Name)
        if ($freeGb -lt 30) {
            Write-Host "          WARNING BireLate needs about 22 GB here: ~3.7 GB of runtimes and ~15 GB of"
            Write-Host "          models on first use, plus ~2.6 GB for HunyuanOCR. 30 GB leaves some room."
        }
    } catch {
        Write-Host ("disk    : could not read free space ({0})" -f $_.Exception.Message)
    }

    # Downloads nest ~200 characters below this folder and Windows stops at 260,
    # which otherwise fails only after gigabytes. OneDrive would upload ~20 GB.
    if ($BireLatePathRoot.Length -gt 55) {
        Write-Host ("folder  : WARNING this folder's path is {0} characters; keep it to 55 or fewer" -f $BireLatePathRoot.Length)
        Write-Host "          (for example C:\Apps\BireLate) or later downloads can fail on Windows' path limit."
    }
    $oneDrive = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial | Where-Object {
        $_ -and ($BireLatePathRoot + '\').StartsWith($_.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) })
    if ($oneDrive.Count -gt 0 -or $BireLatePathRoot -match '\\OneDrive[^\\]*(\\|$)') {
        Write-Host "folder  : WARNING this folder is inside OneDrive, which would try to upload ~20 GB of"
        Write-Host "          models and can lock files mid-download. Move it outside, e.g. C:\Apps\BireLate."
    }

    # Developer Edition and Nightly register under their own names.
    $ff = @()
    foreach ($hive in @("HKLM:\SOFTWARE", "HKCU:\SOFTWARE", "HKLM:\SOFTWARE\WOW6432Node")) {
        foreach ($name in @("Mozilla Firefox", "Mozilla Firefox ESR", "Firefox Developer Edition", "Nightly")) {
            $v = (Get-ItemProperty -Path "$hive\Mozilla\$name" -ErrorAction SilentlyContinue).CurrentVersion
            if ($v) { $ff += [pscustomobject]@{ Name = $name; Version = $v; Major = (($v -split '\.')[0] -as [int]) } }
        }
    }
    $ff = @($ff | Sort-Object Name -Unique)
    if ($ff.Count -eq 0) {
        Write-Host "firefox : WARNING no installed Firefox found in the registry (a portable copy does not show"
        Write-Host "          here). The BireLate extension needs Firefox 142 or newer."
    }
    foreach ($f in $ff) { Write-Host ("firefox : {0} {1}" -f $f.Name, $f.Version) }
    if ($ff.Count -gt 0 -and -not ($ff | Where-Object { $_.Major -ge 142 })) {
        Write-Host "          WARNING too old; the BireLate extension needs Firefox 142 or newer."
    }

    return $ok
}

# ---- downloads ----------------------------------------------------------------

function Format-Size([long]$Bytes) {
    if ($Bytes -lt 1MB) { return "{0:N0} KB" -f [Math]::Ceiling($Bytes / 1KB) }
    return "{0:N0} MB" -f ($Bytes / 1MB)
}

function Test-Pinned([string]$Path, $Pin) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $Pin.Size) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Pin.Sha256
}

# Makes $Dest hold the pinned file: kept if it already verifies, otherwise
# downloaded into local\downloads\<name>.part (resuming an earlier one),
# verified, and only then moved into place.
function Get-Pinned($Pin, [string]$Dest) {
    if (Test-Pinned $Dest $Pin) { Write-Host ("ok      : {0}" -f $Pin.Name); return }
    if ($DryRun) {
        Write-Host ("download: {0} ({1}) -> {2}" -f $Pin.Name, (Format-Size $Pin.Size), $Dest)
        return
    }
    New-Item -ItemType Directory -Path $Downloads, (Split-Path -Parent $Dest) -Force | Out-Null
    $part = Join-Path $Downloads ($Pin.Name + ".part")
    $have = 0
    if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
    if ($have -lt $Pin.Size) {
        Write-Host ("download: {0} ({1})" -f $Pin.Name, (Format-Size $Pin.Size))
        $ErrorActionPreference = "Continue"   # curl's progress meter is on stderr
        & $Curl -L --fail --retry 3 --retry-delay 5 -C - -o $part $Pin.Url
        $code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($code -eq 33 -or $code -eq 36) {
            # The server would not resume (range ignored / bad resume): a kept
            # .part would fail the same way on every run, so start over next time.
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            throw ("downloading {0} failed (curl exit {1}: could not resume); the partial file was deleted" -f $Pin.Name, $code)
        }
        if ($code -ne 0) {
            throw ("downloading {0} failed (curl exit {1})" -f $Pin.Name, $code)
        }
    }
    if (-not (Test-Pinned $part $Pin)) {
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        throw ("{0} did not match its pinned size and SHA256, so it was deleted" -f $Pin.Name)
    }
    Move-Item -LiteralPath $part -Destination $Dest -Force
    Write-Host ("ok      : {0}" -f $Pin.Name)
}

# Extracts $Zip into $Dest, or only the entry $Only (by its file name). A zip
# whose entries all sit under one top folder is flattened, so the files land
# directly in $Dest.
function Expand-Zip([string]$Zip, [string]$Dest, [string]$Only = "") {
    $root = [System.IO.Path]::GetFullPath($Dest).TrimEnd('\') + '\'
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $entries = @($archive.Entries | Where-Object { $_.Name })   # files, not folders
        if ($Only) {
            $entries = @($entries | Where-Object { $_.FullName -eq $Only })
            if ($entries.Count -ne 1) { throw "$Only is not in $Zip" }
        }
        $strip = 0
        $tops = @($entries | ForEach-Object { ($_.FullName -split '/')[0] } | Select-Object -Unique)
        if (-not $Only -and $tops.Count -eq 1 -and -not ($entries | Where-Object { $_.FullName -notmatch '/' })) {
            $strip = $tops[0].Length + 1
        }
        foreach ($e in $entries) {
            $rel = if ($Only) { $e.Name } else { $e.FullName.Substring($strip) }
            $target = [System.IO.Path]::GetFullPath((Join-Path $root $rel))
            if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                throw "refusing to extract $($e.FullName) outside $Dest"
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $target, $true)
        }
    } finally {
        $archive.Dispose()
    }
}

# Installs directory $Dest from one or more archives. A marker records the
# archives that built it, so a re-run skips a complete tree and rebuilds a
# partial one -- or one whose $KeyFile has gone (an antivirus quarantine).
# The tree is assembled beside $Dest and moved in at the end.
function Install-Tree($Archives, [string]$Dest, [string]$Label, [string]$KeyFile) {
    $marker = Join-Path $Dest ".birelate-installed"
    $want = ($Archives | ForEach-Object { $_.Sha256 }) -join " "
    if ((Test-Path -LiteralPath $KeyFile) -and (Test-Path -LiteralPath $marker) -and
            (Get-Content -LiteralPath $marker -Raw).Trim() -eq $want) {
        Write-Host ("ok      : {0}" -f $Label)
        return
    }
    $zips = @(foreach ($a in $Archives) { $z = Join-Path $Downloads $a.Name; Get-Pinned $a $z; $z })
    if ($DryRun) { Write-Host ("extract : {0} -> {1}" -f $Label, $Dest); return }
    $localRoot = Join-Path $BireLatePathRoot "local\"
    if (-not ([System.IO.Path]::GetFullPath($Dest)).StartsWith($localRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to replace ${Dest}: it is outside $localRoot"
    }
    $staging = "$Dest.partial"
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
    foreach ($z in $zips) {
        Write-Host ("extract : {0}" -f (Split-Path -Leaf $z))
        Expand-Zip $z $staging
    }
    [System.IO.File]::WriteAllText((Join-Path $staging ".birelate-installed"), $want)
    if (Test-Path -LiteralPath $Dest) { Remove-Item -LiteralPath $Dest -Recurse -Force }
    Move-Item -LiteralPath $staging -Destination $Dest
    foreach ($z in $zips) { Remove-Item -LiteralPath $z -Force }
    Write-Host ("ok      : {0}" -f $Label)
}

# ---- steps ----------------------------------------------------------------------

function Install-Cudart {
    $pin = $Pins.cudartDll
    if (Test-Pinned $CudartPath $pin) { Write-Host ("ok      : {0}" -f $CudartPath); return }
    $whl = Join-Path $Downloads $Pins.cudart.Name
    Get-Pinned $Pins.cudart $whl
    if ($DryRun) { Write-Host ("extract : {0} -> {1}" -f $pin.Entry, $CudartPath); return }
    # Extracted beside its destination, verified, then moved in: an overwrite in
    # place would write through a hard link to whatever file it points at.
    $staging = "$CudartPath.partial"
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
    Expand-Zip $whl $staging -Only $pin.Entry
    $staged = Join-Path $staging $pin.Name
    if (-not (Test-Pinned $staged $pin)) {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
        throw "cudart64_13.dll from the wheel did not match its pinned SHA256"
    }
    if (Test-Path -LiteralPath $CudartPath) { Remove-Item -LiteralPath $CudartPath -Force }
    Move-Item -LiteralPath $staged -Destination $CudartPath
    Remove-Item -LiteralPath $staging -Recurse -Force
    Remove-Item -LiteralPath $whl -Force
    Write-Host ("ok      : {0}" -f $CudartPath)
}

# Returns $true when HunyuanOCR is to be installed. Asks once: an acceptance is
# recorded in local\state and not asked again.
function Request-HunyuanLicense {
    $record = Join-Path $State "hunyuan-license-accepted.txt"
    if ($SkipHunyuan) {
        Write-Host "hunyuan : skipped (-SkipHunyuan); the main OCR engine is not installed."
        Write-Host "          BireLate will use PaddleOCR-VL: lower quality, especially for"
        Write-Host "          manhua and manhwa."
        return $false
    }
    if (Test-Path -LiteralPath $record) {
        Write-Host "hunyuan : licence accepted earlier ($record)"
        return $true
    }
    Write-Host ""
    Write-Host "HunyuanOCR licence"
    Write-Host "------------------"
    Write-Host "BireLate's default OCR engine is HunyuanOCR, distributed by Tencent under the"
    Write-Host "Tencent Hunyuan Community License Agreement. That licence grants NO rights in"
    Write-Host "the European Union, the United Kingdom or South Korea, and this includes using"
    Write-Host "its outputs there. Read it before accepting:"
    Write-Host "  $LicenseUrl"
    Write-Host "The model files are a community GGUF conversion of it. The converter's page"
    Write-Host "labels them Apache-2.0, but as a derivative of HunyuanOCR they remain under"
    Write-Host "Tencent's licence above."
    Write-Host "BireLate is not affiliated with Tencent. Accepting downloads about 2.6 GB."
    Write-Host ""
    Write-Host "YES installs HunyuanOCR, BireLate's main OCR engine."
    Write-Host "NO means the main OCR engine is NOT installed. Nothing is downloaded and"
    Write-Host "BireLate reads text with PaddleOCR-VL instead (Apache-2.0; downloaded"
    Write-Host "automatically on first use). This LOWERS QUALITY, especially for manhua and"
    Write-Host "manhwa (Chinese and Korean comics). You can run Setup.bat again later to"
    Write-Host "install HunyuanOCR."
    Write-Host ""
    $how = "-AcceptHunyuanLicense"
    if (-not $AcceptHunyuanLicense) {
        if ($DryRun) {
            Write-Host "dry run : a real run asks here; the plan below assumes YES"
            return $true
        }
        $answer = ""
        try { $answer = Read-Host "Type YES to accept and install HunyuanOCR, or press Enter for NO" } catch {}
        if ("$answer".Trim() -ne "YES") {
            Write-Host "hunyuan : NO - the main OCR engine is not installed; nothing downloaded."
            Write-Host "          BireLate will use PaddleOCR-VL: lower quality, especially for"
            Write-Host "          manhua and manhwa. Run Setup.bat again to install HunyuanOCR."
            return $false
        }
        $how = "typed YES at the setup prompt"
    }
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $State -Force | Out-Null
        Set-Content -LiteralPath $record -Encoding Ascii -Value @(
            "HunyuanOCR: Tencent Hunyuan Community License Agreement accepted.",
            ("date    : {0}" -f (Get-Date).ToString("o")),
            ("licence : {0}" -f $LicenseUrl),
            ("how     : {0}" -f $how))
    }
    return $true
}

function Install-Hunyuan {
    Get-Pinned $Pins.model    $Hunyuan.model
    Get-Pinned $Pins.mmproj   $Hunyuan.mmproj
    Get-Pinned $Pins.template $Hunyuan.template
    Get-Pinned $Pins.license  (Join-Path (Get-BireLatePath hunyuan_gguf) "LICENSE")
    Install-Tree @($Pins.llama, $Pins.llamaCudart) (Split-Path -Parent $Hunyuan.llama_server) "llama.cpp b10502 (CUDA 13.3)" $Hunyuan.llama_server
    # The embeddable Python puts its own folder on sys.path (python312._pth),
    # so Pillow unzipped beside python.exe is importable without pip.
    Install-Tree @($Pins.python, $Pins.pillow) (Split-Path -Parent $Hunyuan.python) "Python 3.12.10 + Pillow 12.3.0 (for the OCR shim)" $Hunyuan.python
    if ($DryRun) { return }
    $ErrorActionPreference = "Continue"
    $out = & $Hunyuan.python -c "import PIL, ctypes, http.server; print(PIL.__version__)" 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0 -or "$out".Trim() -ne "12.3.0") {
        throw ("the OCR Python at {0} cannot import what the shim needs: {1}" -f $Hunyuan.python, "$out")
    }
    Write-Host "ok      : OCR Python imports Pillow $out"
}

# ---- main -------------------------------------------------------------------------

Write-Host "BireLate setup in $BireLatePathRoot"
if ($DryRun) { Write-Host "dry run : nothing will be downloaded or changed" }
Write-Host ""

if (-not (Test-Prerequisites)) {
    if (-not $DryRun) {
        Write-Host ""
        Write-Host "Setup stopped. Fix the problem above, then run Setup.bat again."
        exit 1
    }
    Write-Host "dry run : a real run stops here"
}

$withHunyuan = $false
$inHunyuan = $false
$nativeHost = "skipped (-NoNativeHost)"
try {
    Write-Host ""
    Install-Cudart
    # Before the optional HunyuanOCR step, so a failed download there cannot
    # leave the Start button unregistered.
    if (-not $NoNativeHost) {
        Write-Host ""
        & (Join-Path $PSScriptRoot "register-native-host.ps1") -DryRun:$DryRun
        $nativeHost = if ($DryRun) { "would register birelate.server" } else { "registered as birelate.server" }
    }
    $inHunyuan = $true
    $withHunyuan = Request-HunyuanLicense
    if ($withHunyuan) { Install-Hunyuan }
} catch {
    Write-Host ""
    Write-Host ("FAILED  : {0}" -f $_.Exception.Message)
    Write-Host "          Everything already verified is kept; run Setup.bat again to continue"
    Write-Host "          (an interrupted download resumes)."
    if ($inHunyuan) {
        Write-Host "          Or run Setup.bat -SkipHunyuan to use PaddleOCR-VL instead."
    }
    exit 1
}

$hunyuanMissing = @($Hunyuan.Values | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($DryRun) {
    $ocr = if ($withHunyuan) { "HunyuanOCR 1.5, once the downloads above are done" } else { "PaddleOCR-VL 1.6" }
} elseif ($hunyuanMissing.Count -eq 0) {
    $ocr = "HunyuanOCR 1.5 (the default)"
} else {
    $ocr = "PaddleOCR-VL 1.6 (HunyuanOCR is not installed; run Setup.bat again to add it)"
}
Write-Host ""
Write-Host "--- summary"
Write-Host ("cuda    : {0}" -f $CudartPath)
Write-Host ("ocr     : {0}" -f $ocr)
Write-Host ("native  : {0}" -f $nativeHost)
Write-Host ""
if ($DryRun) {
    Write-Host "Dry run finished: nothing was downloaded or changed."
    exit 0
}
Write-Host "Next:"
Write-Host "  1. Double-click 'Start BireLate.bat'. The FIRST start downloads about 3.7 GB"
Write-Host "     of runtimes (CUDA, PyTorch, llama.cpp), and the first translation about"
Write-Host "     15 GB of models, into $(Get-BireLatePath packages)."
Write-Host "     Progress shows in the server window; later starts are quick."
Write-Host "  2. Install the extension: download"
Write-Host "     BireLate-extension-<version>.xpi from the BireLate"
Write-Host "     Releases page; see INSTALL-FIREFOX.txt in this folder."
exit 0
