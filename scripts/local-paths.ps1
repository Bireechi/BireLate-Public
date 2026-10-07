# One manifest shared with local_paths.py. Paths are relative to this checkout,
# never the caller's working directory.
$BireLatePathRoot = Split-Path -Parent $PSScriptRoot
function Get-BireLatePath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $pathConfig = Get-Content -LiteralPath (Join-Path $BireLatePathRoot 'config\local-paths.json') -Raw | ConvertFrom-Json
    $entry = $pathConfig.PSObject.Properties[$Name]
    if (-not $entry -or [string]::IsNullOrWhiteSpace($entry.Value)) {
        throw "Unknown or empty BireLate path: $Name"
    }
    $value = [string]$entry.Value
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $BireLatePathRoot $value }
    return [IO.Path]::GetFullPath($value)
}

# The five files serve.ps1 needs to run HunyuanOCR; setup.ps1 installs them.
# One list, so setup's summary and serve's decision cannot disagree.
function Get-HunyuanFiles {
    $gguf = Get-BireLatePath hunyuan_gguf
    return [ordered]@{
        llama_server = Join-Path (Get-BireLatePath llama_ocr) "llama-server.exe"
        model        = Join-Path $gguf "HunyuanOCR.BF16.gguf"
        mmproj       = Join-Path $gguf "HunyuanOCR.mmproj-bf16.gguf"
        template     = Get-BireLatePath hunyuan_template
        python       = Get-BireLatePath python_ocr
    }
}

# The server token, shared by serve.ps1 and the native host: created once in
# local\state and reused, so the extension can fetch it instead of asking for a
# paste. Writes nothing to stdout -- the native host's stdout is its wire.
# -NoCreate only reads: $null when there is no usable token yet.
function Get-BireLateToken {
    param([switch]$NoCreate)
    $file = Join-Path (Get-BireLatePath state) "popup-token.json"
    if (Test-Path -LiteralPath $file) {
        try {
            $t = (Get-Content -LiteralPath $file -Raw | ConvertFrom-Json).token
            if ($t) { return [string]$t }
        } catch {}
    }
    if ($NoCreate) { return $null }
    New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
    $t = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
    @{ token = $t } | ConvertTo-Json -Compress | Set-Content -LiteralPath $file -Encoding Ascii
    return $t
}
