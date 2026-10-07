# Shared helpers for the scripts that RUN a Koharu binary, such as serve.ps1.
# Dot-source it:
#
#     . "$PSScriptRoot\koharu-runtime.ps1"
#
# Two jobs, both of which were previously copy-pasted into every runner:
#   Resolve-KoharuBinary   pick target\release or target\debug
#   Add-KoharuRuntimePath  put local\runtime\cuda-bootstrap's cudart on PATH
#
# NB on parameter names. $Input is an automatic variable (the pipeline
# enumerator) and a parameter of that name silently arrives empty. $Profile is
# the path to the user's PowerShell profile, and $Debug/$Verbose are common
# parameters. None of them are safe to reuse, hence -Name and -BuildProfile.

. "$PSScriptRoot\local-paths.ps1"
$KoharuTargetDir = Join-Path (Get-BireLatePath koharu_root) "target"
$KoharuBuildScript = Join-Path $PSScriptRoot "build-koharu.ps1"

function Resolve-KoharuBinary {
    <#
      Returns [pscustomobject]{ Path; BuildProfile } for $Name, or $null after
      printing where it looked and how to build it.

      "auto" prefers release and falls back to debug. A debug binary is slow but
      correct, and refusing to start merely because nobody has spent the release
      build yet would be obstructive -- but the fallback is always announced,
      because "everything is unoptimized" is the first thing to suspect when a
      page feels sluggish. Upstream only forces opt-level 3 on the image codecs,
      so in a debug build the renderer, the text auto-fit search, the full-page
      pixel loops and our own code all run at opt-level 0.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [ValidateSet("auto", "release", "debug")][string]$BuildProfile = "auto",
        # Appended to the suggested build command, e.g. "-p birelate-server".
        [string]$BuildArgs = ""
    )

    # `.exe` IS OPTIONAL IN $Name. A cargo bin TARGET has no extension
    # (`birelate-server`, `run`, `paddle_ocr_vl`), so naming the target is the
    # natural call, and a bare `Join-Path` would miss a binary that IS built.
    # The quiet half is what makes it matter: the caller reads "not built",
    # concludes there is nothing to check, and loses the STALE warning below --
    # and this function is the authority on binary freshness.
    #
    # Unconditional rather than platform-gated: this project is Windows-only,
    # and on any other host the extra Test-Path simply misses.
    $candidates = @($Name)
    if (-not [System.IO.Path]::GetExtension($Name)) { $candidates += "$Name.exe" }

    $releaseTried = @($candidates | ForEach-Object { Join-Path $KoharuTargetDir "release\$_" })
    $debugTried   = @($candidates | ForEach-Object { Join-Path $KoharuTargetDir "debug\$_" })
    $release = $releaseTried | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $debug   = $debugTried   | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

    $path  = $null
    $found = $null
    switch ($BuildProfile) {
        "release" {
            if ($release) { $path = $release; $found = "release" }
        }
        "debug" {
            if ($debug) { $path = $debug; $found = "debug" }
        }
        default {
            if ($release) {
                $path = $release; $found = "release"
            } elseif ($debug) {
                $path = $debug; $found = "debug"
                Write-Host "profile : no release build found, falling back to debug (unoptimized, expect it to be slow)"
                Write-Host ("          for the fast one: powershell -File `"{0}`" build --release {1}" -f $KoharuBuildScript, $BuildArgs)
            }
        }
    }

    if (-not $path) {
        Write-Host ("not built: {0}" -f $Name)
        # One line per spelling actually probed.
        if ($BuildProfile -ne "debug")   { foreach ($p in $releaseTried) { Write-Host ("looked in: {0}" -f $p) } }
        if ($BuildProfile -ne "release") { foreach ($p in $debugTried)   { Write-Host ("looked in: {0}" -f $p) } }
        $flag = if ($BuildProfile -eq "debug") { "" } else { " --release" }
        Write-Host ("build   : powershell -File `"{0}`" build{1} {2}" -f $KoharuBuildScript, $flag, $BuildArgs)
        return $null
    }

    # STALENESS IS A WARNING, NOT A REFUSAL, and it is here because the resolver
    # above asks Test-Path and nothing else, never whether the binary is NEWER
    # than the source that produced it. Without it, anything run through this
    # resolver will happily use a release build from before the change you are
    # trying to measure, and report numbers for the old code with no sign
    # anything is wrong.
    #
    # Not a refusal, deliberately: a debug binary is announced and used rather than
    # refused two doors up, for the same reason -- being obstructive at start-up
    # costs more than it saves, and a rebuild forced in the middle of a measurement
    # is its own way to lose an hour. Loud is enough.
    # THE SWEEP IS SCOPED TO THE NAMED BINARY'S OWN CRATES. The newest source
    # ANYWHERE under crates\ is the right scope for birelate-server.exe (its
    # closure is the whole workspace) and wrong for every narrow bin: a
    # server-side edit would false-STALE paddle_ocr_vl.exe. The scope is the
    # workspace-internal dependency closure of the bin target named by
    # $Name -- normal AND build edges, because the -sys crates reach
    # koharu-bindgen only through [build-dependencies]; never dev edges -- plus
    # the root Cargo.toml and Cargo.lock.
    # On ANY scoping failure (cargo missing, locked, slow, unknown bin, missing
    # dir) it falls back to the whole-crates sweep and prints a note line:
    # the failure direction must stay over-firing, never silent under-firing.
    #
    # THE BANNER TEXT BELOW IS LOAD-BEARING for anything that parses this
    # output: the bare uppercase STALE marks the warning, and the line prefixes
    # -- including "note    : " with exactly four spaces -- are fixed. Do not
    # reword either.
    try {
        # Derived from $KoharuTargetDir rather than hardcoded, so the two cannot
        # drift apart; there is no $KoharuDir in this module.
        $koharuRoot = Split-Path -Parent $KoharuTargetDir
        $cratesDir  = Join-Path $koharuRoot "crates"

        $scopedDirs = $null
        try {
            # -replace, never .Trim: Trim is a char-SET trim, and
            # "translate.exe".Trim(".exe") returns "translat".
            $binName = $Name -replace '\.exe$', ''

            # --frozen --offline so cargo can never rewrite Cargo.lock, and a
            # hard 5 s wall so a package-cache lock held by a concurrent build
            # cannot stall a start-up.
            # stdout only -- stderr merged in would break the JSON parse.
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = "cargo"
            $psi.Arguments = 'metadata --no-deps --format-version 1 --frozen --offline --manifest-path "{0}"' -f (Join-Path $koharuRoot "Cargo.toml")
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
            $stderrTask = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit(5000)) {
                try { $proc.Kill() } catch { }
                throw "cargo metadata exceeded 5 s"
            }
            if ($proc.ExitCode -ne 0) { throw ("cargo metadata exited {0}" -f $proc.ExitCode) }
            $meta = $stdoutTask.Result | ConvertFrom-Json

            $byName = @{}
            foreach ($pkg in $meta.packages) { $byName[$pkg.name] = $pkg }
            $owner = $null
            foreach ($pkg in $meta.packages) {
                foreach ($t in $pkg.targets) {
                    if (($t.kind -contains "bin") -and ($t.name -eq $binName)) { $owner = $pkg; break }
                }
                if ($owner) { break }
            }
            if (-not $owner) { throw ("no workspace bin target named {0}" -f $binName) }

            $closure = @{}
            $pending = New-Object System.Collections.ArrayList
            [void]$pending.Add($owner.name)
            while ($pending.Count -gt 0) {
                $current = $pending[0]
                $pending.RemoveAt(0)
                if ($closure.ContainsKey($current)) { continue }
                $closure[$current] = $true
                foreach ($dep in $byName[$current].dependencies) {
                    # dev edges never feed the binary; normal (null kind) and
                    # build edges both do.
                    if (($dep.kind -ne "dev") -and $byName.ContainsKey($dep.name) -and -not $closure.ContainsKey($dep.name)) {
                        [void]$pending.Add($dep.name)
                    }
                }
            }

            $dirs = @()
            foreach ($pkgName in $closure.Keys) { $dirs += Split-Path -Parent $byName[$pkgName].manifest_path }
            $missing = @($dirs | Where-Object { -not (Test-Path -LiteralPath $_) })
            if ($missing.Count -gt 0) { throw ("closure directory missing: {0}" -f $missing[0]) }
            $scopedDirs = $dirs
        } catch {
            # Fall back to the whole-crates sweep rather than narrow silently.
            # First line of the message only, and the uppercase STALE token
            # creplaced away, so a note line can never be mistaken for the
            # STALE banner.
            $reason = (($_.Exception.Message -split "`r?`n")[0]) -creplace "STALE", "stale"
            Write-Host ("note    : could not scope the source sweep for {0} ({1}); sweeping every crate" -f $Name, $reason)
            $scopedDirs = $null
        }

        if ($scopedDirs) {
            # -LiteralPath (a bracket in the install path is not a wildcard), and
            # so the extension filter is a Where-Object: PS 5.1 ignores -Include
            # under -LiteralPath.
            $candidates = @(Get-ChildItem -LiteralPath $scopedDirs -Recurse -File -ErrorAction SilentlyContinue |
                                Where-Object { $_.Extension -eq ".rs" -or $_.Extension -eq ".toml" })
            # The root manifest and lock file feed every binary.
            foreach ($rootFile in @((Join-Path $koharuRoot "Cargo.toml"), (Join-Path $koharuRoot "Cargo.lock"))) {
                if (Test-Path -LiteralPath $rootFile) { $candidates += Get-Item -LiteralPath $rootFile }
            }
        } else {
            $candidates = @(Get-ChildItem -LiteralPath $cratesDir -Recurse -File -ErrorAction SilentlyContinue |
                                Where-Object { $_.Extension -eq ".rs" -or $_.Extension -eq ".toml" })
        }
        $newest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $built = (Get-Item -LiteralPath $path).LastWriteTime
        if ($newest -and $newest.LastWriteTime -gt $built) {
            $age = [int]($newest.LastWriteTime - $built).TotalMinutes
            Write-Host ""
            Write-Host ("STALE   : this {0} binary is OLDER than the source." -f $found)
            Write-Host ("          built {0}, and {1} was edited {2} min later." -f `
                $built.ToString("yyyy-MM-dd HH:mm"), $newest.Name, $age)
            Write-Host ("          Anything measured now describes the OLD code.")
            Write-Host ("          rebuild: powershell -File `"{0}`" build{1} {2}" -f `
                $KoharuBuildScript, $(if ($found -eq "debug") { "" } else { " --release" }), $BuildArgs)
            Write-Host ""
        }
    } catch {
        # A failed staleness check must never stop a run that would otherwise work.
        Write-Host ("note    : could not check binary staleness ({0})" -f $_.Exception.Message)
    }

    return [pscustomobject]@{ Path = $path; BuildProfile = $found }
}

function Add-KoharuRuntimePath {
    <#
      Puts this folder's local\runtime\cuda-bootstrap (setup.ps1 installs
      cudart64_13.dll there) at the front of PATH. Returns $true when
      cudart64_13.dll is findable on PATH afterwards, $false after printing
      "run Setup.bat". Nothing outside this folder is added: another
      application's CUDA directory can carry a foreign cuDNN 9, which breaks
      cuDNN's by-name sub-library loads (serve.ps1 strips those from PATH).
    #>
    $boot = Get-BireLatePath cuda_bootstrap
    if (Test-Path -LiteralPath $boot) { $env:PATH = "$boot;$env:PATH" }

    # Report the cudart we ACTUALLY made findable, not a path we hope exists. Only
    # this one DLL has to be on PATH: Hardware::discover() probes for it by name via
    # cudarc before any package is activated, and if the probe cannot find it the
    # server PANICS at startup (exit 101) -- even with --cpu, it does not fall back.
    # Everything else Koharu preloads by absolute path.
    $cudart = ($env:PATH -split ';' | Where-Object {
        $_ -and (Test-Path -LiteralPath (Join-Path $_ "cudart64_13.dll") -ErrorAction SilentlyContinue)
    } | Select-Object -First 1)
    if ($cudart) {
        Write-Host ("cuda    : cudart64_13.dll from {0}" -f $cudart)
        return $true
    }
    Write-Host "cuda    : cudart64_13.dll is missing, so the server cannot start"
    Write-Host ("          run Setup.bat to install it into {0}" -f $boot)
    return $false
}
