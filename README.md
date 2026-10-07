# BireLate

BireLate is a Firefox extension plus a small local server that translates the
text in manga, manhua and manhwa images in place: it finds the text, reads it,
erases it and letters the translation back onto the page. Everything runs on
your own PC and your own NVIDIA GPU.

What it is not: there is no cloud service, no account, no API key and no
subscription. Nothing you read is sent anywhere except to the server running on
your own machine.

This repository is BireLate's source code. To use BireLate, download two files of
the same version from this repository's **Releases** page:

- `BireLate-<version>-win64.zip` is the server and its setup, ready to run.
  Everything below assumes it.
- `BireLate-extension-<version>.xpi` is the Firefox extension (see
  [Load the extension](#load-the-extension)).

Each release also has `SHA256SUMS.txt`, the SHA256 checksums of both files. To
build the server from this repository instead, see `BUILDING.md`, then follow
this README.

Model weights are not included in either download. They are downloaded from
their publishers the first time they are needed (sizes below).

## Built on Koharu

BireLate is built on **[Koharu](https://koharu.rs)**, the open-source,
ML-powered manga translator written in Rust by Mayo Takanashi (mayocream),
Wangchong Zhou and the Koharu contributors
(<https://github.com/mayocream/koharu>, now <https://github.com/koharu-rs/koharu>).
Koharu does the heavy lifting: text detection, OCR, inpainting, translation with
a local LLM and lettering, plus the runtime that downloads and loads the models.
Most of the models BireLate downloads (the text detector, the LaMa inpainter,
manga-ocr and the font catalog) come from the Koharu project's Hugging Face
repositories.

**BireLate's Koharu is heavily modified.** BireLate forked Koharu at commit
`583ae52` (GPL-3.0) and has changed it extensively since: more
than 30,000 changed lines across some 67 of its source files, touching nearly
every stage of the pipeline:

- OCR engine routing, including HunyuanOCR through a local sidecar, and
  re-reads of rotated, clipped and cut text;
- text detection and the erase mask, including sound-effect handling;
- lettering, layout and text fitting;
- translation prompting with context from earlier pages, glossaries, response
  repair and truncation detection;
- pinned model downloads, model unloading and VRAM handling.

The fork does not take upstream updates, so it also lacks everything Koharu has
changed since (including Koharu's later relicensing; this copy stays GPL-3.0).
BireLate's results, settings and behaviour therefore differ substantially from
Koharu's, and Koharu's documentation does not describe BireLate. On top of the
modified pipeline, BireLate adds the Firefox extension, the local server and the
setup scripts. The changes are summarised in `THIRD-PARTY-NOTICES.md`; the full
modified source is the `koharu\` folder of this repository.

BireLate is an independent, unofficial build: it is not affiliated with or
endorsed by the Koharu project, so please report BireLate problems to this
repository, not to Koharu. For Koharu itself, see <https://koharu.rs>.

The in-page workflow (hover an image, click to translate) is inspired by Torii
Image Translator.

---

## Requirements

- **Windows 10 or 11, 64-bit.** Windows PowerShell 5.1 and `curl.exe` are part
  of Windows (curl since Windows 10 version 1803). PowerShell 7 is not needed.
- **An NVIDIA GPU with driver R580 or newer** (`nvidia-smi` must report CUDA 13.0
  or higher). CUDA 13 no longer supports GPUs older than the RTX 20 / GTX 16
  series, so those are very likely not usable.
- **VRAM.** The default settings need about **21 GiB of free VRAM** for the
  pipeline and the translator. HunyuanOCR, if you install it, holds **about 3 GB
  more for as long as the server runs**, so in practice about **24 GB must be
  free**. BireLate has only been tested on a 32 GB RTX 5090. 24 GB cards are
  untested and probably too tight with HunyuanOCR; smaller cards need a smaller
  translator (see [Low VRAM and alternatives](#low-vram-and-alternatives)).
- **Visual C++ 2015-2022 x64 Redistributable, version 14.40 or newer**
  (<https://aka.ms/vs/17/release/vc_redist.x64.exe>). Setup checks the version.
  Install the latest one even if an older one is already present (many games
  install an older one): with an older runtime the server crashes as it starts.
- **Smart App Control off** (Windows 11). Smart App Control blocks unsigned
  programs such as `bin\birelate-server.exe` and the `llama-server.exe` that
  Setup downloads, and offers no "Run anyway", so BireLate cannot run while it
  is on. See **Windows Security > App & browser control > Smart App Control**.
  Switching it off may not be reversible: some Windows versions only let you
  turn it back on by resetting or reinstalling Windows.
- **Firefox 142 or newer.**
- **About 30 GB of free disk** on the drive that holds the BireLate folder, plus
  about 2 GB of temporary space in `%TEMP%` during the first server start.
- **Internet access** for the downloads: during setup, at the first server
  start, at the first translation, and the first time you use any other engine.

## Install

1. Right-click the downloaded `BireLate-<version>-win64.zip`, choose
   **Properties** and tick **Unblock** (if it is there) before extracting.
   Otherwise Windows may ask about every script it runs.
2. Extract `BireLate-<version>-win64.zip` to a short folder you own, for
   example `C:\Apps`. The zip holds a single folder,
   `BireLate-<version>-win64`; **that folder is "the BireLate folder"**
   everywhere below (here `C:\Apps\BireLate-<version>-win64`). Keep its full
   path under about 55 characters and free of unusual characters: the files it
   downloads sit up to about 200 characters deeper inside it, and Windows' path
   limit is 260. Do not use `C:\Program Files`: setup and the server write into
   a `local\` folder inside the BireLate folder, so it must be writable without
   administrator rights. The `.xpi` can go anywhere; see
   [Load the extension](#load-the-extension).
3. The executables and scripts are not code-signed, so Windows SmartScreen may
   warn about them. Only choose **More info > Run anyway** if you trust where
   the zip came from. Smart App Control, if it is on, blocks them with no such
   choice; see [Requirements](#requirements).

If you move the folder later, run `Setup.bat` again (the browser connection
stores the folder's path).

**Opening a Command Prompt in the folder.** A few steps below type a command
instead of double-clicking. To get a Command Prompt in the BireLate folder, open
the folder in File Explorer, click the address bar, type `cmd` and press Enter.
Use a Command Prompt, not a PowerShell 7 window: the scripts run in Windows
PowerShell 5.1, which loses some of its commands when started from PowerShell 7.

## Run Setup.bat

Double-click `Setup.bat`. It is safe to run again at any time.

**What it checks.** 64-bit Windows, the Visual C++ runtime (14.40 or newer) and
`curl.exe` (setup stops if one is missing or too old); the NVIDIA GPU, driver version and VRAM, free disk
space and the Firefox version (warnings only).

**What it always installs.**

| What | From | Size | Goes to |
|---|---|---|---|
| `cudart64_13.dll`, taken from the NVIDIA CUDA runtime 13.0.48 wheel | PyPI | 3 MB download | `local\runtime\cuda-bootstrap` |

The server's start-up GPU check needs this DLL; without it the server cannot
start.

**The browser connection.** Setup registers the *native messaging host* that
lets the extension's **Start server** button launch the server. This writes one
per-user registry key, `HKCU\Software\Mozilla\NativeMessagingHosts\birelate.server`,
and a manifest file in `local\state`. Restart Firefox afterwards; if you had
already loaded the extension as a temporary add-on, load it again after the
restart. Pass `-NoNativeHost` to skip it; you then start the server with
`Start BireLate.bat`.

**The HunyuanOCR question.** BireLate's default text reader (OCR) is HunyuanOCR,
which Tencent distributes under the *Tencent Hunyuan Community License
Agreement*. That licence grants **no rights in the European Union, the United
Kingdom or South Korea**, including for its output. Setup shows the licence link
and asks:

- **Type `YES`**: the acceptance is recorded in
  `local\state\hunyuan-license-accepted.txt` (you are not asked again), the files
  below are downloaded, and HunyuanOCR becomes the OCR engine. A copy of
  Tencent's licence text is saved next to the model as
  `local\models\hunyuan-gguf\LICENSE`.
- **Press Enter (NO)**: **the main OCR engine is not installed, which lowers
  quality, especially for manhua and manhwa** (Chinese and Korean comics).
  Nothing is downloaded and BireLate reads text with PaddleOCR-VL (Apache-2.0)
  instead, which the server downloads by itself at the first translation. You
  can run `Setup.bat` again later to add HunyuanOCR; then restart the server
  and click **Clear cache** (see [Settings that matter](#settings-that-matter)).

Downloaded only after YES (about 2.6 GB in total, the figure Setup's question
gives; the sizes below are as Setup prints them, where 1 MB is 1,048,576 bytes):

| What | From | Size | Goes to |
|---|---|---|---|
| llama.cpp b10502, CUDA 13.3 build (two zips) | github.com/ggml-org/llama.cpp | 140 MB + 373 MB | `local\runtime\llama-ocr` |
| HunyuanOCR 1.5 GGUF model + vision projector (BF16) | huggingface.co/prithivMLmods/HunyuanOCR-1.5-GGUF-Updated | 1,033 MB + 951 MB | `local\models\hunyuan-gguf` |
| Chat template | huggingface.co/tencent/HunyuanOCR | 1 KB | `local\models\hunyuan-gguf` |
| Tencent's licence text, `LICENSE` | huggingface.co/tencent/HunyuanOCR | 16 KB | `local\models\hunyuan-gguf` |
| Embeddable Python 3.12.10 + Pillow 12.3.0 (runs a small helper) | python.org, PyPI | 11 MB + 7 MB | `local\runtime\python` |

The model files are a community GGUF conversion of HunyuanOCR. Its page labels
them Apache-2.0, but as a derivative of HunyuanOCR they stay under Tencent's
licence.

Every file is pinned to an exact size and SHA256 (and the Hugging Face files to
an exact commit). A file that already verifies is not downloaded again, an
interrupted download resumes on the next run (partial files wait in
`local\downloads`), and a failed run keeps everything already verified.

**Options** (typed after `Setup.bat` in a Command Prompt in the BireLate folder,
see [Install](#install); for example `Setup.bat -DryRun`):

| Option | Effect |
|---|---|
| `-AcceptHunyuanLicense` | accept the HunyuanOCR licence without being asked |
| `-SkipHunyuan` | do not install HunyuanOCR; use PaddleOCR-VL (the same as answering NO: lower quality, especially for manhua and manhwa) |
| `-NoNativeHost` | do not register the browser connection |
| `-DryRun` | print the plan; download and change nothing |

## Load the extension

The extension is not in the BireLate folder. It is
`BireLate-extension-<version>.xpi`, the second download from the **Releases**
page; save it in any folder and keep it there, because a temporary add-on is
loaded from it again each time. `INSTALL-FIREFOX.txt`, in the BireLate folder
and in this repository, gives the methods below step by step. Run `Setup.bat`
and restart Firefox first, so that the **Start server** button works.

The extension is not signed by Mozilla, and release Firefox only installs signed
extensions permanently. Pick one of these:

**a) Temporary add-on (any Firefox).** Open `about:debugging`, click
**This Firefox > Load Temporary Add-on...** and choose
`BireLate-extension-<version>.xpi` (Firefox accepts the `.xpi` directly). It is
removed whenever Firefox closes (including the restart Setup asks for), and you
load it again each time. To keep its settings and cached translations between
loads, set both `extensions.webextensions.keepStorageOnUninstall` and
`extensions.webextensions.keepUuidOnUninstall` to `true` in `about:config`.

**b) Install the .xpi (Firefox Developer Edition, Nightly, ESR or an unbranded
build).** In `about:config` set `xpinstall.signatures.required` to `false`. Then
open `about:addons`, click the gear icon, choose **Install Add-on From File...**
and pick `BireLate-extension-<version>.xpi`. Release and Beta Firefox ignore
that preference.

**c) Sign it yourself (release Firefox).** With a free addons.mozilla.org
account, create API credentials at
<https://addons.mozilla.org/developers/addon/api/key/>. Extract the `.xpi` into
a folder (it is a zip file: copy it, rename the copy to end in `.zip`, then
**Extract All**), or use the `extension\` folder of this repository. Then, with
Node.js installed, run in a Command Prompt (keep the quotes: the folder's path may contain spaces):

```
npx web-ext sign --channel=unlisted --source-dir "<that folder>" --api-key <your JWT issuer> --api-secret <your JWT secret>
```

and install the signed `.xpi` it writes into `web-ext-artifacts\` through
**Install Add-on From File...**. Untested with BireLate. Add-on IDs are unique on
addons.mozilla.org, so if `birelate@local` is already taken, change the `id` in
that folder's `manifest.json` **and** `$ExtensionId` in
`scripts\register-native-host.ps1` in the BireLate folder to the same new value,
then run `Setup.bat` again, or the Start button will not be allowed to talk to
the server launcher.

From a copy of this repository, which has no `.xpi`: for a), choose
`extension\manifest.json` instead; for b), zip the contents of `extension\`,
with `manifest.json` at the top of the zip, and rename it to `.xpi`.

The BireLate toolbar button starts out inside Firefox's **Extensions** menu (the
puzzle-piece icon on the toolbar). Pin it to the toolbar from there so the popup
is one click away.

## Start translating

### 1. Start the server

Either click **Start server** in the popup (Home tab), or double-click
`Start BireLate.bat`. Both open a console window running the server; closing
that window stops the server (and HunyuanOCR with it).

- A server started with the **Start server** button also stops by itself when
  Firefox closes. One started from `Start BireLate.bat` runs until you close its
  window or click **Stop & clear** in the popup.
- **The first start downloads about 3.7 GB of runtimes** before the server
  answers: NVIDIA CUDA 13 libraries including cuDNN (about 1.47 GB, PyPI),
  libtorch 2.12.1 for CUDA 13.0 (about 1.93 GB, download.pytorch.org), and
  llama.cpp plus stable-diffusion.cpp builds (about 0.34 GB, Koharu's GitHub
  releases). They go to `local\runtime\koharu-packages`. Progress is printed in
  the server window. Later starts reuse them.
- The popup shows **Connected to http://127.0.0.1:8765** once the server is up,
  and **No BireLate server at ...** until then, including during that first
  download. A second **Start server** click while the first start is still
  downloading is ignored: it starts no second server, and the popup says *A
  BireLate server is already running or starting - see its window.*
- **Then click Warm up** on the Home tab. The first time, this downloads the
  models (about 15 GB, see below) and loads them, so the long wait happens now
  rather than on the first page you open. Without it the first page waits for
  the download, and with auto-translate on, the images queued behind it give up
  after 10 minutes. Wait until the Home tab shows the models as loaded before
  opening pages: images sent during the warm-up queue behind it. The popup
  stops refreshing after a few minutes, long before a first download is done;
  close and reopen it to see the progress.

### 2. Translate an image

1. Open a page with manga images, open the popup and tick **Enable BireLate on
   this page** (this is remembered per site). If the reader is inside a frame
   from another site, use **Settings > Enable on every page** instead.
2. Hover over an image. Images smaller than 150 px are ignored (change this in
   **Settings > Minimum image detection size**). A small bar appears in the
   image's top-left corner (**Settings > Button hover location**). Its buttons:
   - **BireLate logo**: translate the image in place. Click it again to show the
     original.
   - **↻**: retranslate this image, asking for a different draw.
   - **▦**: edit the detection boxes (add text the detector missed, delete
     false boxes) and retranslate.
   - **A**: auto-translate images as they appear (the same as **Auto-translate
     new images** in the popup).
3. Or right-click any image and choose **Translate image with BireLate** (works
   even on sites you have not enabled) or **Retry translation (new draw)**.

**The first translation (or the first Warm up) downloads about 15 GB of models**
into `local\runtime\koharu-packages`, which takes a long time; watch the server
window. From huggingface.co, each pinned to an exact commit:

| What | Repository | Size |
|---|---|---|
| Text detector (RF-DETR) | mayocream/koharu-layout-rfdetr-seg-2xl-1152 | 161 MB |
| Inpainter (LaMa) | mayocream/lama-manga | 204 MB |
| Translator (Gemma 4 26B-A4B, `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`) | unsloth/gemma-4-26B-A4B-it-qat-GGUF | 14.25 GB |
| Fonts (default lettering font CCWildWords) | mayocream/fonts | 7 MB |

Downloaded only when needed: **PaddleOCR-VL 1.6** (1.93 GB) when HunyuanOCR is
not installed, and also for some Korean re-reads and some Japanese paged pages;
**manga-ocr** (463 MB) for Japanese paged manga sites, which the extension
switches to automatically; and any other engine you pick in the popup.

The models are unloaded from VRAM after 5 minutes without a translation; the
next page loads them again from `local\`. The popup's Home tab
shows what is loaded and the countdown. **Warm up** loads the models ahead of
time, **Free VRAM now** unloads them, and **Stop & clear** (click it twice)
stops the server and deletes the cached translations. HunyuanOCR's ~3 GB is not
released by the idle unload or by **Free VRAM now**; it is freed only when the
server stops.

Other things on the Home tab: **Story context** keeps names consistent across
the pages of one series (on by default; **Reset story context** starts the
series over). **Glossary** pins how terms are translated, one `source =
translation` per line, for the current series and browser session. **This site**
shows the detected layout (manga or webtoon) and language and lets you correct
them for that site. **Keep the artwork** letters the translation over the
untouched page instead of erasing the original text first. If the translate
button never appears, use **Debug > Diagnose this page**.

## Settings that matter

- **OCR.** HunyuanOCR is the default and reads all scripts. On Japanese paged
  manga sites the extension switches to Manga OCR by itself. Choosing an engine
  in this list by hand **turns that automatic choice off for every site**, and
  there is no switch to turn it back on: only removing the extension while
  `extensions.webextensions.keepStorageOnUninstall` is off (Firefox's default)
  clears it, together with all other settings and the cache. To correct a
  single site, use the **This site** card instead. If HunyuanOCR is not
  installed, then while this list is on HunyuanOCR the popup says *"HunyuanOCR
  is not installed on this server; paddleocr-vl-1.6 is used instead. Run
  Setup.bat to add it."* To add it later, run `Setup.bat`, restart the server
  (it checks for HunyuanOCR only when it starts), then click **Clear cache**.
- **Target language.** English by default; Spanish, French, German, Portuguese
  (BR), Russian, Chinese (Simplified) and Korean are offered.
- **Inpainter.** LaMa by default. The others download their own models on first
  use; their sizes and VRAM needs are not documented here.
- **Translation backend / Model.** Must match how the server was started. With
  the defaults leave them on **Built-in local engine** and blank. A mismatch is
  refused with an error that names the server's setting.
- **Clear cache** (Cache tab). Translations are cached for 24 hours (limits in
  **Settings > Cache**). **If you add HunyuanOCR later**, run `Setup.bat`,
  restart the server, **then click Clear cache**: pages cached before were read
  by PaddleOCR-VL and would otherwise be shown again unchanged.
- **Settings > Server.** The server URL (default `http://127.0.0.1:8765`) and the
  shared token. The token is filled in automatically through the browser
  connection; see Troubleshooting if it is not.

## Low VRAM and alternatives

**None of the options in this section have been tested in this package.** They
are server parameters, given to `Start BireLate.bat` from a Command Prompt in
the BireLate folder (see [Install](#install); note the quotes, the file name has
a space):

```
"Start BireLate.bat" -Llm gemma4-12b-it
```

The popup's **Start server** button always starts the server with the default
settings, so use the batch file (or a shortcut to it with arguments) when you
want parameters.

| Parameter | What it does |
|---|---|
| `-Llm <id>` | Use a different local translator. Ids that exist include `gemma4-e2b-it`, `gemma4-e4b-it`, `gemma4-12b-it`, `gemma4-31b-it`, `qwen3.5-9b`, `ministral-3-8b-instruct` and `ministral-3-14b-instruct`. Only the default and `gemma4-31b-it` are pinned to an exact revision. The others are not: each time the server loads one (the first page after a start or after an idle unload) it asks huggingface.co for the current revision, so it needs internet access then even when the files are already on disk, and it downloads the whole model again whenever its repository changes. Translation quality with them is untested. Leave the popup's **Model** blank or set it to the same id. |
| `-ColdReserveBytes <bytes>` | How much free VRAM the server requires before loading the models for a page; below it the page is refused with "not enough VRAM" instead of crashing. Default 21 GiB (22548578304) for the local translator, 4 GiB with `-Provider ollama`. Lower it for a smaller `-Llm`; raise it if a larger one is admitted and then crashes. |
| `-Provider ollama -Llm <tag>` | Translate with a model served by Ollama (default endpoint `http://localhost:11434/v1`) instead of the built-in engine, so the server itself needs much less VRAM (Ollama's model needs its own). Set the popup's **Translation backend** to **Ollama** and **Model** to the tag (or blank). |
| `-BaseUrl <url>` | With `-Provider ollama`, use another OpenAI-compatible server, e.g. `-BaseUrl http://127.0.0.1:1234/v1`. |
| `-IdleUnloadSecs <n>` | Unload the models after `n` idle seconds (default 300); `0` keeps them loaded. |
| `-DryRun` | Print what would run (binary, token file, OCR choice, the full argument list) and start nothing. |

When installed, HunyuanOCR still holds its ~3 GB for as long as the server runs.
To run without it, delete
`local\models\hunyuan-gguf`; the server then uses PaddleOCR-VL. Setup will
download it again on its next run unless you pass `-SkipHunyuan` (or delete
`local\state\hunyuan-license-accepted.txt` to be asked again).

## Troubleshooting

| Symptom | What to do |
|---|---|
| Server window: *cudart64_13.dll is missing, so the server cannot start* | Run `Setup.bat`. |
| Server window: *missing : ...\bin\birelate-server.exe* or *...koharu-torch.dll* | Extract the win64 zip again, and check your antivirus quarantine. |
| Windows reports a missing `VCRUNTIME` DLL, and the server window says *server exited with code ...* and waits for a key | Install the Visual C++ Redistributable (see Requirements), then run `Setup.bat` again: it checks for the runtime and does not continue without it. |
| Server window: *server exited with code -1073741819* (0xC0000005, an access violation), often with nothing useful printed above it | Usually a Visual C++ runtime older than 14.40. Install the latest Visual C++ Redistributable (see Requirements) even if one is already installed, then run `Setup.bat` again. |
| Windows says Smart App Control blocked `birelate-server.exe` or `llama-server.exe` | Smart App Control must be off; see Requirements. |
| Popup: *Not enough VRAM: X free, Y needed.*; a translation fails with *insufficient VRAM* (HTTP 507) | Something else is using the GPU. Close games and other GPU or AI programs, then try again. HunyuanOCR's ~3 GB counts as used. See also [Low VRAM and alternatives](#low-vram-and-alternatives). |
| Server window: *path : removed ... (a foreign cuDNN 9 would clash with Koharu's)* | Information only. Another CUDA program's folder was hidden from this server's PATH; nothing else is changed. |
| Server window: *failed to bind 127.0.0.1:8765* | Another BireLate server, or another program, is using port 8765. Close the other server window. To use another port: `"Start BireLate.bat" -Addr 127.0.0.1:8766` and set **Settings > BireLate server URL** to `http://127.0.0.1:8766` (untested; the Start button always uses 8765). |
| Popup: *No BireLate server at ...* | The server is not running, or it is still downloading on its first start. Look at the server window. |
| Requests fail with *401* (token) | The extension normally gets the token through the browser connection. Without it (`-NoNativeHost`), copy the `token` value from `local\state\popup-token.json` into **Settings > Shared token**. |
| Start button: *no launcher registered - run Setup.bat once, then restart Firefox* | Run `Setup.bat` (without `-NoNativeHost`) and restart Firefox. Also needed after moving the folder. |
| Server window: *sidecar : llama-server DIED during load* or *never came up inside 240s* | HunyuanOCR failed to start, often for lack of VRAM. Its log is `%TEMP%\birelate-llama-server.log` (and `.log.err`). To run without HunyuanOCR, see the end of the previous section. |
| The translate button never appears | Tick **Enable BireLate on this page**, reload the tab, check the minimum image size, then use **Debug > Diagnose this page**. |
| A server started from `Start BireLate.bat` stopped when Firefox closed | Known issue: if you used the **Start server** button earlier in the same Firefox session, closing Firefox can also stop a server started later from the batch file. Start it again from the batch file. |
| You want to see what would run | `"Start BireLate.bat" -DryRun` (server) or `Setup.bat -DryRun` (setup). Neither downloads or starts anything. |

## Where things are stored, and uninstalling

Inside the BireLate folder, in `local\`:

| Folder | Contents |
|---|---|
| `local\runtime\koharu-packages` | runtimes and models the server downloads (the bulk of the space) |
| `local\runtime\cuda-bootstrap` | `cudart64_13.dll` |
| `local\runtime\llama-ocr`, `local\runtime\python`, `local\models\hunyuan-gguf` | HunyuanOCR and what runs it |
| `local\downloads` | unfinished setup downloads |
| `local\state` | the server token, the HunyuanOCR acceptance record, the browser-connection manifest and the browser-close marker |

Outside the folder:

- **Registry:** `HKCU\Software\Mozilla\NativeMessagingHosts\birelate.server`,
  written by `Setup.bat` unless `-NoNativeHost`.
- **`%TEMP%`:** `birelate-llama-server.log` and `birelate-llama-server.log.err`
  (HunyuanOCR's log, rewritten at each start); a `koharu-storage-*` folder per
  translation, which holds the page being translated and is deleted when the
  request ends (leftovers from a crash are deleted at the next server start);
  and, during the first start, the runtime archives, which are deleted after
  unpacking.
- **Firefox profile:** the extension's settings and its translation cache. They
  are removed with the extension (unless you set
  `extensions.webextensions.keepStorageOnUninstall`).

To uninstall:

1. In the popup, click **Stop & clear** (twice) to stop the server and delete
   the cached translations; if the server is not running, click **Clear cache**
   on the Cache tab instead.
2. Reset the `about:config` preferences you changed for loading the extension
   (`extensions.webextensions.keepStorageOnUninstall`,
   `extensions.webextensions.keepUuidOnUninstall`,
   `xpinstall.signatures.required`). Do this before the next step: while
   `keepStorageOnUninstall` is `true`, removing the extension leaves its
   settings and cache behind in the Firefox profile.
3. Remove the extension in `about:addons` (or `about:debugging`).
4. In a Command Prompt in the BireLate folder (see [Install](#install)), run
   `powershell -ExecutionPolicy Bypass -File scripts\register-native-host.ps1 -Unregister`.
5. Delete the BireLate folder.
6. Optionally delete the two `birelate-llama-server.log*` files and any
   `koharu-storage-*` folders in `%TEMP%`.

## Privacy

- All detection, OCR, inpainting and translation run on your PC.
- The network is used only for the downloads listed above, from
  huggingface.co, pypi.org / files.pythonhosted.org, download.pytorch.org,
  github.com and python.org. Downloads happen on first use; the downloaded files
  are reused afterwards.
- GitHub and Hugging Face answer a download by redirecting it to their own file
  servers, so a firewall allowlist must let those through as well: for GitHub
  release files, hosts such as `release-assets.githubusercontent.com` or
  `objects.githubusercontent.com`; for Hugging Face, its CDN and Xet storage
  hosts, such as `cdn-lfs` or `cdn.hf.co` names (one observed was
  `us.aws.cdn.hf.co`). These are examples, not a complete list: the services
  choose the host, and it can depend on your region and change over time.
- The extension takes images from the page you are viewing and sends them only
  to the local server at `127.0.0.1:8765`. By default the server listens only on
  127.0.0.1, requires a token, and rejects requests coming from web pages.

## Licences

- BireLate is licensed under the **GNU GPL v3** (see `LICENSE`, in this
  repository and in the win64 zip). It contains a
  modified copy of **Koharu** (<https://github.com/mayocream/koharu>) at commit
  583ae52, which is GPL-3.0 at that commit; a few of its runtime crates are
  MIT OR Apache-2.0.
- The Rust crates compiled into `bin\birelate-server.exe` and their licence
  texts are listed in `THIRD-PARTY-RUST.txt`. `bin\koharu-torch.dll` contains
  code from the PyTorch (libtorch) headers, under PyTorch's BSD-style licence,
  reproduced in `THIRD-PARTY-NOTICES.md`.
- The components and models downloaded at setup and first use are not part of
  the release downloads; each comes from its publisher under its own licence,
  summarised in `THIRD-PARTY-NOTICES.md`. Points worth knowing before you use them:
  - **HunyuanOCR**: Tencent Hunyuan Community License Agreement. No rights in
    the European Union, the United Kingdom or South Korea, including for its
    output; Tencent's acceptable use policy applies. Tencent is not affiliated
    with, associated with, sponsoring or endorsing BireLate. Notice:

    > Tencent Hunyuan is licensed under the Tencent Hunyuan Community License
    > Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark
    > rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.

  - **Gemma 4** (the translator): Apache-2.0 per Google's Gemma 4 licence page,
    which also links Google's prohibited use policy.
  - **The text detector** was fine-tuned on Manga109, whose images are for
    academic and non-commercial use; its model card leaves compliance to the
    user. **manga-ocr** was trained on Manga109-s.
  - **The default lettering font, CCWildWords**, comes from the mayocream/fonts
    dataset, which states no licence. It is downloaded at runtime, not shipped.
  - **NVIDIA CUDA and cuDNN** are under NVIDIA's licence agreements.

## Source code

The GPL gives you the right to the source. This repository is that source: all
of BireLate's and the modified Koharu's, namely the server, the modified Koharu,
the scripts, the native messaging host, the extension and its tests. Each
release is tagged, and the tag holds the source of that release's downloads;
GitHub also offers it as a download on the release's page. The extension's
`.xpi` also contains the extension's complete source. The Rust crates they
depend on (from crates.io) and the CPU build of libtorch that `koharu-torch.dll`
is compiled against (from download.pytorch.org) are fetched at build time. To
build, see `BUILDING.md`. A few files of upstream Koharu that are not build
inputs (its website docs, benchmark images and similar) are not in this
repository; see `THIRD-PARTY-NOTICES.md`.
