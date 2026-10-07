# Start birelate-server with BireLate's default settings baked in, plus the
# HunyuanOCR sidecar when that engine is installed. "Start BireLate.bat" and the
# extension's Start button both run this; closing its window stops everything it
# started. -DryRun prints what would run and starts nothing.
#
# NO FILE LOGGING ON THIS PATH: this script, the sidecar it starts and the
# server itself print to the console only, and the window closes with a clean
# stop. Whoever wants a record redirects THIS script's output; do not add a file
# appender, transcript or tee here.
#
# Uses bin\birelate-server.exe when it exists (the prebuilt download). Otherwise
# it runs a source build from koharu\target, preferring release and falling back
# to debug with a warning; -BuildProfile pins one explicitly. The provider is
# FIXED for the life of the process: switching the local LLM at runtime loads the
# new weights before dropping the old ones, and nothing evicts them, so a 32 GB
# card aborts. A request asking for a different backend gets a 409 explaining
# that. Under -Provider ollama the model name is free to vary per request,
# because those weights live in Ollama's process rather than ours.
#
# NB: do not name a parameter $Input. It is a PowerShell automatic variable and
# silently arrives empty. $Profile and $Debug are taken too, hence -BuildProfile.
param(
    [string]$Addr         = "127.0.0.1:8765",
    [string]$Token        = "",
    # Must track cli.rs's own --ocr default. This file bakes the settings in
    # deliberately (see the header), so the copy is the design -- but a drift
    # is invisible: the launchers below pass no -Ocr, and the extension
    # overrides per request, so the only symptom is a pipeline reload on the
    # first page of every session. (The extension picks manga-ocr only on
    # evidence that the material is Japanese AND paged.) This is the
    # READER-facing launcher, so it must match the extension or the first
    # translate of every session forces a reload.
    [string]$Ocr          = "hunyuan-ocr-1.5",
    [string]$Inpainting   = "lama",
    [string]$Provider     = "local",
    # Must track cli.rs's DEFAULT_LOCAL_MODEL, for the same reason -Ocr must
    # track --ocr: this file bakes the settings in on purpose, and the copy is
    # only safe while it agrees. The MoE matched gemma4-31b-it on quality across
    # four comparisons, runs the page in 0.69x the time and holds 13.26 GB of
    # weights against 16.09. -Llm gemma4-31b-it selects the dense model instead.
    [string]$Llm          = "gemma4-26b-a4b-it",
    [string]$Language     = "en-US",
    [string]$Instructions = "",
    [string]$BaseUrl      = "",
    [string[]]$FontFamily = @(),
    [string[]]$AllowOrigin = @(),
    [string[]]$AllowHost  = @(),
    [int]$MaxUploadBytes  = 0,
    [int]$RequestTimeoutSecs = 0,
    # -1 means "say nothing and let the server's own default stand". 0 is a real
    # value that disables the idle unload, so it cannot double as "unset".
    [int]$IdleUnloadSecs  = -1,
    [ValidateSet("auto", "release", "debug")][string]$BuildProfile = "auto",
    [switch]$Warmup,
    # Letter translated DIALOGUE in capitals, the traditional comic convention.
    # Narration, signs, captions and credits are left alone -- detection already
    # tells them apart, and shouting a caption is wrong. Off by default because
    # it pairs with a comic face and the fallback here is Arial.
    [switch]$UppercaseDialogue,
    # Turns OFF the free-standing caption size cap, so it can be A/B'd against
    # itself. Only sent when asked for, so this still drives an older binary.
    [switch]$NoFitFreeText,
    # How far above the page's prevailing dialogue size one balloon may be set.
    # "" leaves the server's own default; "0" turns the rule off, which is the
    # other arm of the A/B. A string rather than a number so that neither the
    # locale's decimal separator nor "0 might mean unset" can get in the way.
    [string]$SizeCoherence = "",
    # How many source/target pairs the story window carries into the next page's
    # prompt. "" leaves the server's own 96; "0" turns the window off, which is
    # the no-story arm. A string rather than a number for the same reason
    # -SizeCoherence is one: PowerShell reads the number 0 as falsy, so an int
    # would drop the very arm it exists to name.
    [string]$StoryPairs = "",
    [string]$StoryExcludesSfx = "",
    # Route a text region whose larger side reaches this many pixels to
    # PaddleOCR-VL. Deliberately EMPTY rather than "448": the server owns that
    # default, and a copy here could silently drift from it. Pass 0 to turn the
    # routing off.
    [string]$LargeCropOcr = "",
    # The ceiling on that routing, as a fraction of the page's own area. EMPTY
    # for the same reason as above -- the server owns the 0.5. Pass 0 for the
    # no-ceiling arm.
    [string]$LargeCropOcrMaxArea = "",
    # The term the free-text size cap solves for. Empty leaves the
    # measured 0.12 default alone; raising it is the only lever that makes a
    # vertical caption bigger, and it scales with the SQUARE ROOT.
    [string]$SourceInkFraction = "",
    # Turn a tall vertical column into its own ink instead of widening it. ON in
    # the server's own default; pass "false" for the other arm.
    [string]$RotateFreeTextColumns = "",
    # Grow a column the detector truncated at a page edge, and report a
    # sub-floor edge box as an `edge_hints` entry. ON in the server's own
    # default; pass "false" for the other arm.
    #
    # It also lowers the detector's TEXT request to 0.20, so with it on the reply's
    # `edge_hints` carry the RAW geometry of sub-floor edge boxes -- which is the
    # only committed way to read a refused box's height off the device.
    [string]$RepairClippedColumns = "",
    # Read a tall kana-free free-text column a SECOND time with the crop turned
    # 90 CCW, and prefer the turned read. ON in cli.rs.
    [string]$RereadRotatedColumns = "",
    [string]$OrientationConfidenceMargin = "",
    [string]$PerturbRereadGrowPx = "",
    [string]$UprightPass = "",
    # Read a SYNTHESISED bubble a second time turned 180 degrees, keeping the
    # flipped read only when its confidence clearly wins. ON in cli.rs.
    [string]$FlipRereadBubbles = "",
    # Buy one spotting call on a sparse or decline-carrying page and mint
    # regions for display runs the detector never boxed. ON in cli.rs.
    [string]$SpotRescue = "",
    # Let a shipped spot rescue also join the erase mask. ON in cli.rs; a
    # separate switch from the lettering, so each can be judged on its own.
    [string]$SpotRescueErase = "",
    # A wide scream-read mint becomes the mark-replacement device --
    # ink-scoped erase, one styled gradient replacement on the ink's own axis.
    # Ships ON; "false" is the same-binary control arm.
    [string]$ReplaceScreamMarks = "",
    # Let a seam composite buy the spot call regardless of its region
    # count. Ships ON; "false" is the control arm.
    [string]$SpotRescueJoined = "",
    # Read a detected BUBBLE holding no text region of its own as one. ON in
    # cli.rs.
    [string]$ReadTextlessBubbles = "",
    # The translator prompt's containment sentence -- each segment letters its
    # own source only. OFF in cli.rs; "true"/"false" name the two arms.
    # The wire reports it per page as containment_clause.
    [string]$ContainmentClause = "",
    # Describe each segment to the translator (kind, and whether the
    # artwork strikes it through). OFF in cli.rs; "true" turns it on.
    [string]$SegmentContext = "",
    # Two paired levers. The floor opens a replacement-only score band on
    # JOINED pages; the tie-break lets the column-shaped box of an NMS pair win.
    # BOTH ON in cli.rs (floor 0.20); the tie-break is scoped to declared
    # zh/ko in engine.rs. "0" / "false" are the off arms.
    [string]$JoinedPageTextFloor = "",
    [string]$AxisAwareNms = "",
    # What happens to the box the tie-break above EVICTS. ON in cli.rs -- it
    # hands an evicted box's uncovered residue back as its own region, which is
    # the only way text in that residue reaches OCR at all; "false" drops it.
    [string]$NmsResidueRegions = "",
    # Keep a read the pipeline already refuses to letter OUT of the
    # translation request. ON in cli.rs. On a 179-slice Chinese test chapter,
    # 188 of 331 regions were refused and not one was ever lettered, and that
    # junk re-rolled the good text sharing its request. Never empties a page.
    [string]$SkipUnletteredReads = "",
    # Re-draw an authored strike-through mark over the English that
    # replaces the struck name. ON in cli.rs; not script-scoped.
    [string]$StrikeThroughDevices = "",
    # Free-standing lettering takes fill/weight/outline from the drawn
    # ink's sampled colour. ON in cli.rs; "false" is the same-binary control arm.
    [string]$SampledInkLettering = "",
    # Scope a watermark verdict to the site's own text instead of condemning the
    # whole region. ON in cli.rs.
    [string]$ScopeWatermarkRefusals = "",
    # Paired with -KoreanScriptStrict: a script/punctuation refusal reaches
    # dialogue on a declared zh/ko page, with the same read withdrawn from the
    # erase mask. ON in cli.rs.
    [string]$LeaveMisreadBubbles = "",
    # A declared-Korean read with no hangul is a mismatch. ON in cli.rs.
    [string]$KoreanScriptStrict = "",
    # Buy one reserve re-read for a lever-refused dialogue read on
    # declared ko, admitted iff hangul-majority. ON in cli.rs.
    [string]$RereadRefusedDialogue = "",
    # Offer the free-text column turn on an UNJOINED page too. ON by default.
    [string]$TurnUnjoinedColumns = "",
    # Read a region over that ceiling with no engine at all.
    #
    # A [string], NOT a [switch], and that is load-bearing. `--skip-implausible-regions`
    # is `default_value = "true"` with NO `num_args`, so it REQUIRES a value: passing the
    # bare token makes the server refuse to boot with
    #   error: a value is required for '--skip-implausible-regions <BOOL>'
    # A [switch] could also only ever request `true`, which is already the default, so
    # the `false` arm would be unreachable either way.
    [string]$SkipImplausibleRegions = "",
    # Whether the translator's sliding-window layers allocate the whole context.
    # "" leaves llama.cpp's default; "false" asks for the reduced KV cache.
    [string]$SwaFull = "",
    # Whether a stage is asked for work before its weights are paged in, so a
    # page with nothing to translate does not load the 16.5 GiB LLM. "" leaves
    # the server's default, which is on; "false" is the other arm. A string for
    # the same reason -SwaFull is one, and only sent when non-empty so this
    # script still drives a binary built before the flag existed.
    [string]$SkipEmptyStages = "",
    # Sampling temperature for the translation LLM. "" leaves the catalog's own
    # per-model value, which is NOT shared: both Gemma-4 arms carry Google's
    # recommended 1.0 and ministral-3-14b-instruct carries 0.05.
    #
    # This exists because an offline benchmark pins 0 in `replay.rs`, which
    # never touches the server -- so without it a page rendered through
    # `serve.ps1` is sampled at a DIFFERENT temperature from the benchmark that
    # chose the model, and nothing says so. A string, not a double, for the same
    # reason -StoryPairs is one: PowerShell reads the number 0 as falsy, so an
    # int would silently drop the one value most worth passing.
    [string]$TranslationTemperature = "",

    # ---- the remaining cli.rs flags ---------------------------------------------
    #
    # Every cli.rs flag needs a passthrough here, or its arm cannot be rendered
    # through this launcher at all. Many of these gate pixel-changing behaviour
    # that ships ON, so the "off" arm is the one worth passing.
    #
    # The three shapes below are not stylistic. A [switch] can only ever request
    # `true`; a [string] can request either arm AND leave the server's own default
    # alone when empty. Which one is correct is decided by `cli.rs`, and getting it
    # wrong stops the server booting -- see -SkipImplausibleRegions above.

    # Bare `bool` in cli.rs: these accept NO value, so they are switches. Each is a
    # negative lever whose gate ships ON, so passing it is always the "off" arm.
    [switch]$LetterImplausibleText,
    [switch]$LetterDuplicateText,
    [switch]$NoDuplicateOrientedOverlap,
    # ON server-side; pass "false" for the control arm.
    [string]$DuplicateSharedSource = "",
    [switch]$NoCollisionRelief,
    [switch]$NoTranslateSfx,
    [switch]$RefineTextMask,
    [switch]$InkMask,
    # Negative lever: the anchored lettering for cut balloons ships ON, so
    # passing this is the "off" control arm.
    [switch]$NoEdgeAnchoredLettering,

    # `Option<bool>` with `default_value = "true"` and no `num_args`: these REQUIRE
    # a value. Bare tokens here are the -SkipImplausibleRegions boot failure again,
    # so they are [string] and pass "true"/"false" explicitly.
    [string]$SkipImplausibleMasks = "",
    [string]$WithdrawIllegibleMasks = "",
    [string]$WithdrawUnreadMasks = "",
    [string]$SeamSafeErase = "",
    [string]$ReleaseCachedVram = "",

    # Value-taking. [string] rather than the numeric type for the same reason
    # -StoryPairs is one: PowerShell reads the number 0 as falsy, so an int would
    # silently drop the arm most worth passing -- and 0 is a real value for every
    # one of these.
    [string]$Hyphenation = "",
    [string]$RoremSteps = "",
    [string]$FluxStrength = "",
    [string]$OnomatopoeiaThreshold = "",
    [string]$MaskScale = "",
    [string]$ColdReserveBytes = "",
    # Debug instrument: dump the settled per-region detection masks
    # and the assembled inpaint masks as PNGs into this directory. Empty -- the
    # default, the shipped state -- dumps nothing.
    [string]$DebugMaskDir = "",

    # Repeatable, like -FontFamily. Ships ON: the dictionary holds
    # measured/corrected entries only, built not to guess. Pass
    # -SfxDictionary "" to run bare.
    [string[]]$SfxDictionary = @("$PSScriptRoot\..\data\sfx-dictionary.json"),

    [switch]$Cpu,
    [switch]$NoToken,
    # Resolve everything -- binary, OCR decision, PATH, the full argument list --
    # print it, and exit without starting any process.
    [switch]$DryRun
)

# Cargo writes progress to stderr and PS 5.1 wraps native stderr as
# NativeCommandError, which "Stop" would treat as a fatal error.
$ErrorActionPreference = "Continue"

. "$PSScriptRoot\koharu-runtime.ps1"

# Every exit goes through here. A failure keeps the window open so its message
# can be read -- the extension's Start button spawns this script directly, and a
# missing file exits in milliseconds. "Start BireLate.bat" pauses by itself and
# sets BIRELATE_BAT so the reader is not asked twice.
function Exit-Serve([int]$Code) {
    if ($Code -ne 0 -and -not $env:BIRELATE_BAT) {
        $null = Read-Host "press Enter to close this window"
    }
    exit $Code
}

$prebuilt = Join-Path (Get-BireLatePath bin) "birelate-server.exe"
if (Test-Path -LiteralPath $prebuilt) {
    # The prebuilt server loads its torch bridge from its own directory.
    $torchDll = Join-Path (Split-Path -Parent $prebuilt) "koharu-torch.dll"
    if (-not (Test-Path -LiteralPath $torchDll)) {
        Write-Host "missing : $torchDll"
        Write-Host "          it must sit beside birelate-server.exe -- extract the BireLate zip again"
        Exit-Serve 1
    }
    $binary = [pscustomobject]@{ Path = $prebuilt; BuildProfile = "prebuilt" }
} elseif (-not (Test-Path -LiteralPath (Get-BireLatePath koharu_root))) {
    # The binary download has no koharu\ source tree to build from.
    Write-Host "missing : $prebuilt"
    Write-Host "          extract the BireLate zip again, and check your antivirus quarantine"
    Exit-Serve 1
} else {
    $binary = Resolve-KoharuBinary -Name "birelate-server.exe" -BuildProfile $BuildProfile -BuildArgs "-p birelate-server"
    if (-not $binary) { Exit-Serve 1 }
}
Write-Host ("binary  : {0}" -f $binary.Path)

# The same token the extension's Start button uses, so the extension can fetch it
# through the native host whichever way the server was started. A dry run only
# reads it.
if (-not $Token -and -not $NoToken) {
    $tokenFile = Join-Path (Get-BireLatePath state) "popup-token.json"
    $Token = Get-BireLateToken -NoCreate:$DryRun
    if ($Token) {
        Write-Host ("token   : from {0}" -f $tokenFile)
    } else {
        $Token = "<token: would be created>"
        Write-Host ("token   : {0} at {1}" -f $Token, $tokenFile)
    }
}

# A cuDNN 9.x directory on PATH (another CUDA application's bin) breaks cuDNN's
# by-name sub-library loads with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Koharu
# loads its own cuDNN by absolute path and needs none of them. Process-local only.
$keptPath = @()
foreach ($dir in ($env:PATH -split ';' | Where-Object { $_ })) {
    if (Get-ChildItem -LiteralPath $dir -Filter "cudnn*64_9.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
        Write-Host ("path    : removed {0} (a foreign cuDNN 9 would clash with Koharu's)" -f $dir)
    } else {
        $keptPath += $dir
    }
}
$env:PATH = $keptPath -join ';'

if (-not (Add-KoharuRuntimePath)) {
    if (-not $DryRun) { Exit-Serve 1 }
    Write-Host "dry run : a real run stops here"
}

# Koharu's package store -- CUDA, torch and llama.cpp runtimes plus the models,
# all downloaded on first use -- lives inside this folder, not in %LOCALAPPDATA%.
$store = Get-BireLatePath packages
if (-not $DryRun) { New-Item -ItemType Directory -Path $store -Force | Out-Null }

# The default engine lives OUTSIDE this process: `hunyuan-ocr-1.5` is served by
# llama-server plus a small Python shim on 11436, which the OCR stage probes at
# load. The extension asks for it on every request whatever -Ocr says, so this
# is decided regardless of -Ocr. An already-listening 11436 is used and left
# alone -- whoever started it owns it. Otherwise the sidecar is started here when
# all its files are installed; Setup installs them only if the licence is
# accepted, and without them every request for Hunyuan is served by PaddleOCR-VL.
$substitute = ""
$hunyuan = $null   # the sidecar's files, when this launcher is to start it
$sidecarUp = $false
try {
    Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:11436/api/tags" -TimeoutSec 2 | Out-Null
    $sidecarUp = $true
    Write-Host "sidecar : already listening on 11436, leaving it alone"
} catch {}
if (-not $sidecarUp) {
    $files = Get-HunyuanFiles
    $missing = @($files.Values | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) {
        Write-Host "ocr     : HunyuanOCR is not installed (run Setup.bat to add it); using PaddleOCR-VL instead"
        if ($DryRun) { foreach ($m in $missing) { Write-Host ("          missing: {0}" -f $m) } }
        $substitute = "paddleocr-vl-1.6"
        if ($Ocr -eq "hunyuan-ocr-1.5") { $Ocr = $substitute }
    } else {
        $hunyuan = $files
    }
}

$a = @(
    "--addr", $Addr,
    "--ocr", $Ocr,
    "--inpainting", $Inpainting,
    "--provider", $Provider,
    "--target-language", $Language,
    "--store-dir", $store
)
if ($substitute) { $a += @("--hunyuan-substitute", $substitute) }
if ($Llm)          { $a += @("--llm", $Llm) }
if ($BaseUrl)      { $a += @("--base-url", $BaseUrl) }
if ($Instructions) { $a += @("--translation-instructions", $Instructions) }
if ($Token)        { $a += @("--token", $Token) }
if ($NoToken)      { $a += "--no-token" }
if ($Cpu)          { $a += "--cpu" }
if ($MaxUploadBytes -gt 0)     { $a += @("--max-upload-bytes", $MaxUploadBytes) }
if ($RequestTimeoutSecs -gt 0) { $a += @("--request-timeout-secs", $RequestTimeoutSecs) }
# Both of these are only sent when asked for, so this script still drives a
# server binary built before they existed.
if ($IdleUnloadSecs -ge 0)     { $a += @("--idle-unload-secs", $IdleUnloadSecs) }
if ($Warmup)                   { $a += "--warmup" }
# The no-orphans tie: the server dies with this script, exactly as the sidecar
# stack does. Without it, a kill test saw the sidecars fold in 5.4 s while the
# server kept answering /health, which is the orphan this line ends.
$a += @("--watch-pid", $PID)
if ($UppercaseDialogue)        { $a += "--uppercase-dialogue" }
if ($NoFitFreeText)            { $a += "--no-fit-free-text" }
if ($SizeCoherence)            { $a += @("--size-coherence", $SizeCoherence) }
if ($StoryPairs)               { $a += @("--story-pairs", $StoryPairs) }
if ($StoryExcludesSfx)         { $a += @("--story-excludes-sfx", $StoryExcludesSfx) }
if ($LargeCropOcr)             { $a += @("--large-crop-ocr", $LargeCropOcr) }
if ($LargeCropOcrMaxArea)      { $a += @("--large-crop-ocr-max-area", $LargeCropOcrMaxArea) }
if ($SourceInkFraction)        { $a += @("--source-ink-fraction", $SourceInkFraction) }
if ($RotateFreeTextColumns)    { $a += @("--rotate-free-text-columns", $RotateFreeTextColumns) }
if ($RepairClippedColumns)     { $a += @("--repair-clipped-columns", $RepairClippedColumns) }
if ($RereadRotatedColumns)     { $a += @("--reread-rotated-columns", $RereadRotatedColumns) }
if ($OrientationConfidenceMargin) { $a += @("--orientation-confidence-margin", $OrientationConfidenceMargin) }
if ($UprightPass)              { $a += @("--upright-pass", $UprightPass) }
if ($FlipRereadBubbles)        { $a += @("--flip-reread-bubbles", $FlipRereadBubbles) }
if ($SpotRescue)               { $a += @("--spot-rescue", $SpotRescue) }
if ($SpotRescueErase)          { $a += @("--spot-rescue-erase", $SpotRescueErase) }
if ($ReplaceScreamMarks)       { $a += @("--replace-scream-marks", $ReplaceScreamMarks) }
if ($SpotRescueJoined)         { $a += @("--spot-rescue-joined", $SpotRescueJoined) }
if ($PerturbRereadGrowPx) { $a += @("--perturb-reread-grow-px", $PerturbRereadGrowPx) }
if ($ReadTextlessBubbles)      { $a += @("--read-textless-bubbles", $ReadTextlessBubbles) }
if ($ContainmentClause)        { $a += @("--containment-clause", $ContainmentClause) }
if ($SegmentContext)           { $a += @("--segment-context", $SegmentContext) }
if ($JoinedPageTextFloor)      { $a += @("--joined-page-text-floor", $JoinedPageTextFloor) }
if ($AxisAwareNms)             { $a += @("--axis-aware-nms", $AxisAwareNms) }
if ($NmsResidueRegions)        { $a += @("--nms-residue-regions", $NmsResidueRegions) }
if ($SkipUnletteredReads)      { $a += @("--skip-unlettered-reads", $SkipUnletteredReads) }
if ($StrikeThroughDevices)     { $a += @("--strike-through-devices", $StrikeThroughDevices) }
if ($SampledInkLettering)      { $a += @("--sampled-ink-lettering", $SampledInkLettering) }
if ($ScopeWatermarkRefusals)   { $a += @("--scope-watermark-refusals", $ScopeWatermarkRefusals) }
if ($LeaveMisreadBubbles)      { $a += @("--leave-misread-bubbles", $LeaveMisreadBubbles) }
if ($KoreanScriptStrict)       { $a += @("--korean-script-strict", $KoreanScriptStrict) }
if ($RereadRefusedDialogue)    { $a += @("--reread-refused-dialogue", $RereadRefusedDialogue) }
if ($TurnUnjoinedColumns)      { $a += @("--turn-unjoined-columns", $TurnUnjoinedColumns) }
if ($SkipImplausibleRegions)   { $a += @("--skip-implausible-regions", $SkipImplausibleRegions) }
if ($SwaFull)                  { $a += @("--swa-full", $SwaFull) }
if ($SkipEmptyStages)          { $a += @("--skip-empty-stages", $SkipEmptyStages) }
if ($TranslationTemperature)   { $a += @("--translation-temperature", $TranslationTemperature) }
# The remaining flags. Same three shapes as the parameter block, in the same order.
if ($LetterImplausibleText)      { $a += "--letter-implausible-text" }
if ($LetterDuplicateText)        { $a += "--letter-duplicate-text" }
if ($NoDuplicateOrientedOverlap) { $a += "--no-duplicate-oriented-overlap" }
if ($DuplicateSharedSource)      { $a += @("--duplicate-shared-source", $DuplicateSharedSource) }
if ($NoCollisionRelief)          { $a += "--no-collision-relief" }
if ($NoTranslateSfx)             { $a += "--no-translate-sfx" }
if ($RefineTextMask)             { $a += "--refine-text-mask" }
if ($InkMask)                    { $a += "--ink-mask" }
if ($NoEdgeAnchoredLettering)    { $a += "--no-edge-anchored-lettering" }
if ($SkipImplausibleMasks)   { $a += @("--skip-implausible-masks", $SkipImplausibleMasks) }
if ($WithdrawIllegibleMasks) { $a += @("--withdraw-illegible-masks", $WithdrawIllegibleMasks) }
if ($WithdrawUnreadMasks)    { $a += @("--withdraw-unread-masks", $WithdrawUnreadMasks) }
if ($SeamSafeErase)          { $a += @("--seam-safe-erase", $SeamSafeErase) }
if ($ReleaseCachedVram)      { $a += @("--release-cached-vram", $ReleaseCachedVram) }
if ($Hyphenation)            { $a += @("--hyphenation", $Hyphenation) }
if ($RoremSteps)             { $a += @("--rorem-steps", $RoremSteps) }
if ($FluxStrength)           { $a += @("--flux-strength", $FluxStrength) }
if ($OnomatopoeiaThreshold)  { $a += @("--onomatopoeia-threshold", $OnomatopoeiaThreshold) }
if ($MaskScale)              { $a += @("--mask-scale", $MaskScale) }
if ($ColdReserveBytes)       { $a += @("--cold-reserve-bytes", $ColdReserveBytes) }
if ($DebugMaskDir)           { $a += @("--debug-mask-dir", $DebugMaskDir) }
foreach ($d in ($SfxDictionary | Where-Object { $_ })) { $a += @("--sfx-dictionary", $d) }
foreach ($f in $FontFamily)  { $a += @("--font-family", $f) }
foreach ($o in $AllowOrigin) { $a += @("--allow-origin", $o) }
foreach ($h in $AllowHost)   { $a += @("--allow-host", $h) }

Write-Host "birelate: $Ocr / $Inpainting / ${Provider}:$Llm -> $Language on $Addr"
Write-Host ("profile : {0}" -f $binary.BuildProfile)
# The full argument vector, token redacted, so a stored run's log PROVES which
# flags its server was launched with: a control arm whose expected result is
# "byte-identical" cannot otherwise be told apart from a passthrough that
# silently never delivered the flag.
Write-Host ("args    : {0}" -f (($a | ForEach-Object { if ($_ -eq $Token) { "<token>" } else { $_ } }) -join " "))
if ($IdleUnloadSecs -eq 0) {
    Write-Host "idle    : unload disabled, the models stay resident until the process exits"
} elseif ($IdleUnloadSecs -gt 0) {
    Write-Host ("idle    : unload after {0}s with no translation" -f $IdleUnloadSecs)
}
# -Warmup loads the models at boot, which is real GPU work before the first
# request. Say so, because other programs may be using the GPU.
if ($Warmup) { Write-Host "warmup  : loading models at startup (GPU work before the first request)" }
# The extension and this script both default to "local", so a fresh install
# does not 409. The line below still earns its place, because -Provider is a
# parameter: anyone who overrides it here HAS to move the popup to match, and
# the 409 that follows if they do not names the value without explaining where
# to type it.
Write-Host "popup   : set Provider to '$Provider', and leave Model blank or set it to '$Llm'"

# This launcher owns the sidecar's lifetime exactly as it owns the server's:
# started here when the engine needs it, stopped when the server exits. It holds
# ~3 GB of VRAM for the session.
$sidecar = $null   # the shim serving 11436
$llamaSrv = $null  # the llama-server behind it
if ($hunyuan) {
    $shimPy = Join-Path $PSScriptRoot "hunyuan-llamacpp-shim.py"
    $llamaLog = Join-Path $env:TEMP "birelate-llama-server.log"
    $llamaArgs = @(
        "--model", "`"$($hunyuan.model)`"", "--mmproj", "`"$($hunyuan.mmproj)`"",
        "--port", "11437", "--host", "127.0.0.1", "-ngl", "99",
        "-c", "10240", "--no-webui", "--jinja",
        "--chat-template-file", "`"$($hunyuan.template)`"")
    Write-Host ("sidecar : {0} {1}" -f $hunyuan.llama_server, ($llamaArgs -join " "))
    Write-Host ("          {0} `"{1}`" --port 11436 --llama-url http://127.0.0.1:11437 --watch-pid {2} --llama-pid <llama-server>" -f $hunyuan.python, $shimPy, $PID)
}

if ($DryRun) {
    Write-Host "dry run : nothing started"
    exit 0
}

if ($hunyuan) {
    Write-Host "sidecar : starting llama-server (BF16, ~3 GB, GPU work) + shim on 127.0.0.1:11436"
    $llamaSrv = Start-Process -FilePath $hunyuan.llama_server -ArgumentList $llamaArgs `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $llamaLog -RedirectStandardError "$llamaLog.err"
    $ready = $false
    $deadline = (Get-Date).AddSeconds(240)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if ($llamaSrv.HasExited) { Write-Host "sidecar : llama-server DIED during load -- read $llamaLog"; Exit-Serve 1 }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:11437/health" -TimeoutSec 2 | Out-Null
            $ready = $true; break
        } catch {}
    }
    if (-not $ready) { Write-Host "sidecar : llama-server never came up inside 240s"; Stop-Process -Id $llamaSrv.Id -Force -ErrorAction SilentlyContinue; Exit-Serve 1 }
    # --watch-pid / --llama-pid: the shim exits by itself when THIS process dies
    # (taking llama-server with it) or when its backend has been dead ~40s, so a
    # killed session cannot leave a squatter that answers the probe above and
    # 500s every read. /api/tags is a 503 while the backend is down, so "already
    # listening" means "already listening AND healthy".
    $sidecar = Start-Process -FilePath $hunyuan.python -ArgumentList @(
        "`"$shimPy`"", "--port", "11436", "--llama-url", "http://127.0.0.1:11437",
        "--watch-pid", $PID, "--llama-pid", $llamaSrv.Id) `
        -PassThru -WindowStyle Hidden
    $up = $false
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if ($sidecar.HasExited) { Write-Host "sidecar : shim DIED"; Stop-Process -Id $llamaSrv.Id -Force -ErrorAction SilentlyContinue; Exit-Serve 1 }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:11436/api/tags" -TimeoutSec 2 | Out-Null
            $up = $true; break
        } catch {}
    }
    if (-not $up) { Write-Host "sidecar : shim never came up inside 30s"; Stop-Process -Id $sidecar.Id -Force -ErrorAction SilentlyContinue; Stop-Process -Id $llamaSrv.Id -Force -ErrorAction SilentlyContinue; Exit-Serve 1 }
    Write-Host "sidecar : llama.cpp + shim ready"
}

Write-Host "---"
& $binary.Path @a
$serverExit = $LASTEXITCODE
if ($sidecar -and -not $sidecar.HasExited) {
    Stop-Process -Id $sidecar.Id -Force -ErrorAction SilentlyContinue
    Write-Host "sidecar : stopped with the server"
}
if ($llamaSrv -and -not $llamaSrv.HasExited) {
    Stop-Process -Id $llamaSrv.Id -Force -ErrorAction SilentlyContinue
    Write-Host "sidecar : llama-server stopped with the server"
}
# A clean stop (exit 0) lets the window close with the process; a failure keeps
# it open so the message above can be read.
if ($serverExit -ne 0) {
    Write-Host ("server exited with code {0} -- read the output above." -f $serverExit)
}
Exit-Serve $serverExit
