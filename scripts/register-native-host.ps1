<#
.SYNOPSIS
    Registers the BireLate native messaging host, so the popup's Start button
    can launch the server.

.DESCRIPTION
    Setup.bat runs this for you; -NoNativeHost there skips it. It is the only
    script in BireLate that writes to the registry.

    What it writes: the host manifest, generated at
        local\state\birelate.server.json
    and one key,
        HKCU:\Software\Mozilla\NativeMessagingHosts\birelate.server
    whose default value is the manifest's absolute path. HKCU, not HKLM --
    per-user, no elevation, and it cannot affect anyone else on the machine.
    Moving the BireLate folder breaks the path; run this again afterwards.

    Why it is needed at all: a Firefox extension can stop the server over HTTP,
    but nothing in the WebExtension API can start a process. Native messaging is
    the sanctioned route, and it requires the browser to be told where the host
    lives.

    The host itself only ever launches this folder's own scripts\serve.ps1. It
    accepts no path and no command from the extension, because a registered
    native host is reachable by any extension named in its manifest.

.PARAMETER Unregister
    Removes the key and leaves the files alone. The Start button then reports
    that no launcher is registered, which is also the state before you ever run
    this.

.PARAMETER DryRun
    Print what would be written, and write nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register-native-host.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register-native-host.ps1 -Unregister
#>

param(
    [switch]$Unregister,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\local-paths.ps1"

# NOT "birelate": a development build of BireLate may register that name, and
# this must never overwrite or remove it.
$HostName = "birelate.server"
$ExtensionId = "birelate@local"
$KeyPath = "HKCU:\Software\Mozilla\NativeMessagingHosts\$HostName"
$BatchPath = Join-Path $BireLatePathRoot "native\birelate-host.bat"
$ManifestPath = Join-Path (Get-BireLatePath state) "$HostName.json"

if ($Unregister) {
    if ($DryRun) {
        Write-Host "dry run  : would remove $KeyPath"
    } elseif (Test-Path $KeyPath) {
        Remove-Item -Path $KeyPath -Recurse -Force
        Write-Host "removed  : $KeyPath"
    } else {
        Write-Host "not set  : $KeyPath"
    }
    Write-Host "The Start button will now say no launcher is registered."
    return
}

if (-not (Test-Path -LiteralPath $BatchPath)) {
    throw "missing $BatchPath - extract the full BireLate zip"
}

$json = ConvertTo-Json -Depth 4 -InputObject ([ordered]@{
    name               = $HostName
    description        = "Starts the BireLate translation server for the BireLate extension."
    path               = $BatchPath
    type               = "stdio"
    allowed_extensions = @($ExtensionId)
})

if ($DryRun) {
    Write-Host "dry run  : would write $ManifestPath"
    Write-Host $json
    Write-Host "dry run  : would set $KeyPath (default) = $ManifestPath"
    return
}

# .NET, not Out-File: PS 5.1's utf8 encoding always writes a BOM, and a BOM in
# front of the opening brace makes Firefox reject the manifest as malformed.
New-Item -ItemType Directory -Path (Split-Path -Parent $ManifestPath) -Force | Out-Null
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ManifestPath, $json, $utf8NoBom)

New-Item -Path $KeyPath -Force | Out-Null
Set-ItemProperty -Path $KeyPath -Name "(default)" -Value $ManifestPath

Write-Host "host     : $BatchPath"
Write-Host "manifest : $ManifestPath"
Write-Host "registry : $KeyPath"
Write-Host "Registered. Restart Firefox, then the popup's Start server button will work."
