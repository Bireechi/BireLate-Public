<#
    Native messaging host: the one thing a Firefox extension cannot do itself.

    An extension can stop the server over HTTP, but it cannot *start* a process.
    Firefox's native messaging is the only sanctioned route, so this is the
    smallest host that does the job: it accepts one message, launches serve.ps1
    detached, and answers.

    The wire protocol is not JSON on a line. Firefox writes a 4-byte native-order
    (little-endian on Windows) unsigned length, then that many bytes of UTF-8
    JSON, on stdin; the reply takes the same shape on stdout. Anything else
    written to stdout corrupts the stream, which is why this script prints
    nothing and sends its diagnostics to stderr.

    Only ever launches the one script it was installed beside. It takes no path,
    no arguments and no command from the message -- a native host is reachable by
    any extension listed in its manifest, so accepting a command would turn it
    into a general-purpose process launcher.

    A start that actually launches serve.ps1 also arms a detached sentinel
    (birelate-sentinel.ps1) that waits on the spawning Firefox's process
    handle and POSTs /shutdown when the browser exits, so the stack this
    button started ends with the browser -- clean close and crash alike.
    The token in local\state\popup-token.json is shared with Start
    BireLate.bat, so it cannot tell this host's servers from anyone else's;
    the launcher's contract ("a server it did not start is left alone") is
    kept by never arming the sentinel when a server is already listening.
    Nothing about the sentinel is extension-controlled: no address, token or
    target ever comes from the message.
#>

$ErrorActionPreference = "Stop"

# Binary stdio. Get-Content/Write-Host would apply encoding and line handling to
# a length-prefixed byte stream and desynchronise it on the first message.
$stdin = [Console]::OpenStandardInput()
$stdout = [Console]::OpenStandardOutput()

function Read-Exactly {
    param([int]$Count)
    $buffer = New-Object byte[] $Count
    $read = 0
    while ($read -lt $Count) {
        $got = $stdin.Read($buffer, $read, $Count - $read)
        # 0 is a clean EOF: Firefox closed the pipe, so the host should exit
        # rather than spin.
        if ($got -le 0) { return $null }
        $read += $got
    }
    return $buffer
}

function Send-Reply {
    param([hashtable]$Payload)
    $json = $Payload | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $stdout.Write([BitConverter]::GetBytes([uint32]$bytes.Length), 0, 4)
    $stdout.Write($bytes, 0, $bytes.Length)
    $stdout.Flush()
}

# All state -- the token, the sentinel marker -- lives in this install's
# local\state (native -> ..\local\state), never in %LOCALAPPDATA%,
# so two installs on one machine cannot see each other's. Get-BireLateToken is
# the token serve.ps1 is started with, created once and reused every session:
# the extension stores it after the first start, and a fresh-per-start token
# would 401 every session whose profile kept the old one.
. ([System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\scripts\local-paths.ps1")))
$StateDir = Get-BireLatePath state
$TokenFile = Join-Path $StateDir "popup-token.json"

# The Firefox that spawned this host, found by walking the parent chain
# (firefox.exe -> cmd.exe running the .bat -> this powershell). Start time
# rides along so a reused PID cannot impersonate it later.
function Find-FirefoxAncestor {
    $cursor = $PID
    for ($i = 0; $i -lt 10; $i++) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $cursor" -ErrorAction SilentlyContinue
        if (-not $proc) { return $null }
        if ($proc.Name -ieq "firefox.exe") {
            try {
                $p = Get-Process -Id $proc.ProcessId -ErrorAction Stop
                return @{ ProcessId = [int]$proc.ProcessId; StartTicks = [long]$p.StartTime.Ticks }
            } catch { return $null }
        }
        $cursor = $proc.ParentProcessId
        if (-not $cursor) { return $null }
    }
    return $null
}

# Arm (or re-point) the browser-close sentinel. One sentinel per Firefox: an
# alive one re-reads the marker at fire time, so rewriting the marker is enough
# to hand it a fresh token or address without spawning a twin.
function Start-BrowserWatch {
    param([string]$Addr, [string]$Token)
    $ff = Find-FirefoxAncestor
    # Not spawned by Firefox (a manual run of this script): nothing to watch,
    # and failing open here keeps the start itself working.
    if (-not $ff) { return }
    if (-not (Test-Path -LiteralPath $StateDir)) {
        New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
    }
    $marker = Join-Path $StateDir "browser-watch.json"
    $sentinelPid = $null
    if (Test-Path -LiteralPath $marker) {
        try {
            $old = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
            if ($old.firefoxPid -eq $ff.ProcessId -and $old.sentinelPid) {
                $s = Get-Process -Id $old.sentinelPid -ErrorAction SilentlyContinue
                if ($s) { $sentinelPid = [int]$old.sentinelPid }
            }
        } catch {}
    }
    if (-not $sentinelPid) {
        $sentinel = Join-Path $PSScriptRoot "birelate-sentinel.ps1"
        # SPAWN THROUGH WMI, NOT Start-Process (measured). Firefox runs
        # this host inside a Windows Job object whose handle-close kills every
        # process in it; a Start-Process child inherits the job, so a
        # "detached" sentinel dies the moment the host is reaped -- dead
        # while the browser runs, no POST on close, marker left behind. The
        # identical sentinel spawned by hand with the identical argument line
        # parks and fires correctly, which isolates the spawn environment as
        # the mechanism.
        # Win32_Process.Create runs the child under WmiPrvSE, outside the
        # caller's job, which is the standard breakaway PowerShell can reach
        # without native code.
        $cmd = ('powershell.exe -NoLogo -NoProfile -NonInteractive ' +
            '-ExecutionPolicy Bypass -File "{0}" -MarkerPath "{1}" ' +
            '-WatchPid {2} -WatchStartTicks {3}') -f
            $sentinel, $marker, $ff.ProcessId, $ff.StartTicks
        $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly `
            -Property @{ ShowWindow = [uint16]0 }
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine               = $cmd
            ProcessStartupInformation = $startup
        }
        if ($r -and $r.ReturnValue -eq 0) { $sentinelPid = [int]$r.ProcessId }
    }
    # Whether THIS host sits in a job is the diagnostic that names the kill
    # mechanism above; recorded so it can be read off the marker.
    $inJob = $null
    try {
        if (-not ("BireLate.JobProbe" -as [type])) {
            Add-Type -Namespace BireLate -Name JobProbe -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern bool IsProcessInJob(System.IntPtr hProcess, System.IntPtr hJob, out bool result);
'@
        }
        $flag = $false
        if ([BireLate.JobProbe]::IsProcessInJob(
                [System.Diagnostics.Process]::GetCurrentProcess().Handle,
                [System.IntPtr]::Zero, [ref]$flag)) {
            $inJob = [bool]$flag
        }
    } catch {}
    @{
        firefoxPid        = $ff.ProcessId
        firefoxStartTicks = $ff.StartTicks
        addr              = $Addr
        token             = $Token
        sentinelPid       = $sentinelPid
        hostInJob         = $inJob
    } | ConvertTo-Json -Compress | Set-Content -LiteralPath $marker -Encoding Ascii
}

function Start-BireLateServer {
    $serve = Join-Path $PSScriptRoot "..\scripts\serve.ps1"
    $serve = [System.IO.Path]::GetFullPath($serve)
    if (-not (Test-Path -LiteralPath $serve)) {
        return @{ ok = $false; error = "serve.ps1 not found at $serve" }
    }

    $token = Get-BireLateToken

    # Already up? Starting a second one would bind-fail and, worse, briefly put
    # two processes on the GPU. The sentinel is NOT armed here: the listener
    # may be a server started from Start BireLate.bat, which carries this same
    # token, so arming would let closing Firefox stop a server this host did not
    # start. One this host did start earlier is already watched.
    try {
        $existing = Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction Stop
        if ($existing) {
            return @{ ok = $true; already = $true; token = $token }
        }
    } catch {
        # No listener, or the cmdlet is unavailable. Either way, carry on and let
        # the bind be the authority.
    }

    # Nothing listening is not enough: on the first start the server downloads
    # ~3.7 GB of runtimes before it binds 8765, so a second click would open a
    # second server. A serve.ps1 or birelate-server.exe from THIS folder counts
    # as running, listening or not; the sentinel stays unarmed for the same
    # reason as above.
    $root = $BireLatePathRoot.TrimEnd('\') + '\'
    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue -Filter (
        "Name = 'powershell.exe' OR Name = 'pwsh.exe' OR Name = 'birelate-server.exe'"))
    foreach ($p in $procs) {
        $mine = if ($p.Name -ieq "birelate-server.exe") {
            $p.ExecutablePath -and $p.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)
        } else {
            $p.CommandLine -and $p.CommandLine.IndexOf($serve, [StringComparison]::OrdinalIgnoreCase) -ge 0
        }
        if ($mine) { return @{ ok = $true; already = $true; token = $token } }
    }

    <#
        A visible window on purpose. serve.ps1 holds ~20 GB of VRAM for as
        long as it runs, so its window must be visible and closable. Hidden,
        the only way to end it would be the button that started it.

        SPAWNED THROUGH WMI, NOT Start-Process (measured). Firefox runs this
        host in a kill-on-close Job object (hostInJob:true in the marker is the
        direct evidence), Start-Process children inherit the job, and a
        serve.ps1 started that way dies with the host's reaping. The observed
        signature: `start` replies {already:false, ok:true}, then /health
        never answers and the stack never populates. Same mechanism and same
        fix as the sentinel above; ShowWindow=1 keeps the window visible,
        CurrentDirectory replaces -WorkingDirectory.
    #>
    # NOT -NoExit: a clean shutdown must close this window with it. serve.ps1
    # itself pauses on any non-zero exit, so a failure's message still gets read.
    $serveCmd = ('powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ' +
        '-File "{0}" -Token {1}') -f $serve, $token
    $serveStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly `
        -Property @{ ShowWindow = [uint16]1 }
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine               = $serveCmd
        CurrentDirectory          = (Split-Path $serve -Parent)
        ProcessStartupInformation = $serveStartup
    }
    if (-not $created -or $created.ReturnValue -ne 0) {
        return @{ ok = $false; error = ("serve.ps1 spawn failed, WMI return {0}" -f
            $(if ($created) { $created.ReturnValue } else { "null" })) }
    }
    Start-BrowserWatch -Addr "127.0.0.1:8765" -Token $token
    return @{ ok = $true; already = $false; token = $token }
}

while ($true) {
    $header = Read-Exactly -Count 4
    if ($null -eq $header) { break }
    $length = [BitConverter]::ToUInt32($header, 0)
    # Firefox caps a message at 1 MB; anything near it is not ours.
    if ($length -eq 0 -or $length -gt 65536) { break }

    $body = Read-Exactly -Count ([int]$length)
    if ($null -eq $body) { break }

    try {
        $message = [System.Text.Encoding]::UTF8.GetString($body) | ConvertFrom-Json
        if ($message.command -eq "start") {
            Send-Reply (Start-BireLateServer)
        } elseif ($message.command -eq "token") {
            <#
                Hands back the token serve.ps1 starts the server with, so a
                profile that lost its storage (or never had it) does not need a
                paste. The same file the start command writes. Read-only: a
                token that does not exist yet is created by the first start.
            #>
            if (Test-Path -LiteralPath $TokenFile) {
                try {
                    $token = (Get-Content -LiteralPath $TokenFile -Raw | ConvertFrom-Json).token
                    if ($token) { Send-Reply @{ ok = $true; token = $token } }
                    else { Send-Reply @{ ok = $false; error = "no token recorded yet" } }
                } catch {
                    Send-Reply @{ ok = $false; error = $_.Exception.Message }
                }
            } else {
                Send-Reply @{ ok = $false; error = "no token yet - start the server once" }
            }
        } else {
            Send-Reply @{ ok = $false; error = "unknown command" }
        }
    } catch {
        Send-Reply @{ ok = $false; error = $_.Exception.Message }
    }
}
