<#
    Browser-close sentinel: when the browser closes, the extension's native host
    tells serve.ps1 to shut down, and the watchdog chain does the rest.

    Spawned DETACHED by birelate-host.ps1 when the popup's Start button launches
    serve.ps1. Detached because the host itself is reaped by Firefox as soon as
    the reply is read, and because a per-port EOF handler would miss a browser
    that crashes: this waits on the Firefox PROCESS handle, not on a pipe, so
    clean close and crash look the same to it.

    On Firefox's exit it re-reads the marker written beside it and POSTs
    /shutdown with the token the host started the server with. That token is
    shared with Start BireLate.bat, so it does not tell servers apart: the host
    keeps the contract ("a server it did not start is left alone") by arming
    this only when it launched serve.ps1 itself. A server started with a
    different token answers 401 and is left alone.
    The graceful /shutdown path plus the --watch-pid chain folds server, shim
    and llama-server in ~1.6-5.5 s (measured).

    Writes no log of its own. The marker is state, not history, and is
    deleted once acted on.
#>
param(
    [Parameter(Mandatory = $true)][string]$MarkerPath,
    [Parameter(Mandatory = $true)][int]$WatchPid,
    [Parameter(Mandatory = $true)][long]$WatchStartTicks
)

$ErrorActionPreference = "SilentlyContinue"

function Read-Marker {
    if (-not (Test-Path -LiteralPath $MarkerPath)) { return $null }
    try { return Get-Content -LiteralPath $MarkerPath -Raw | ConvertFrom-Json } catch { return $null }
}

# A PID alone is reusable; the start time makes it this Firefox and not a
# process that later inherited its number. A mismatch means the watched
# Firefox is already gone, which is the fire path, not an error.
$alive = $false
try {
    $proc = Get-Process -Id $WatchPid -ErrorAction Stop
    if ($proc.Name -ieq "firefox" -and $proc.StartTime.Ticks -eq $WatchStartTicks) { $alive = $true }
} catch {}

if ($alive) {
    try { Wait-Process -Id $WatchPid -ErrorAction Stop } catch {}
}

# Superseded? A later Start in a DIFFERENT Firefox rewrote the marker; if that
# browser still runs, its own sentinel owns the shutdown and this one must not
# fire under it.
$marker = Read-Marker
if ($null -eq $marker) { exit 0 }
if ($marker.firefoxPid -ne $WatchPid) {
    try {
        $other = Get-Process -Id $marker.firefoxPid -ErrorAction Stop
        if ($other.Name -ieq "firefox") { exit 0 }
    } catch {}
}

$headers = @{}
if ($marker.token) { $headers["x-koharu-token"] = [string]$marker.token }
try {
    Invoke-WebRequest -Uri ("http://{0}/shutdown" -f $marker.addr) -Method POST `
        -Headers $headers -TimeoutSec 5 -UseBasicParsing | Out-Null
} catch {
    # Connection refused: the server already exited with its window. 401: not
    # our server. Both are the correct end state and neither is retried.
}

try { Remove-Item -LiteralPath $MarkerPath -Force } catch {}
exit 0
