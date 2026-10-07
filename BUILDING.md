# Building BireLate from source

This builds `birelate-server.exe` and `koharu-torch.dll`, the two files the
release download `BireLate-<version>-win64.zip` ships in `bin\`. The extension
and the scripts need no build step.

## Getting the source

This repository is the source. Clone it with git, or download a release's
source from that release on the **Releases** page (GitHub's **Source code
(zip)**). Each release is tagged, and the tag holds the source its downloads
were built from.

## What is in this repository

| Path | What it is |
|---|---|
| `server\` | `birelate-server`, the local HTTP server (Rust, axum) |
| `koharu\` | Koharu (<https://github.com/mayocream/koharu>) at upstream commit 583ae52, with BireLate's modifications. A Cargo workspace; the desktop-app crates are excluded from it. |
| `extension\` | the Firefox extension (plain JavaScript); the release's `.xpi` is these files |
| `native\` | the native messaging host that the Start button uses; `scripts\register-native-host.ps1` registers it |
| `INSTALL-FIREFOX.txt` | how to install the extension, the same file as in the win64 zip |
| `scripts\` | `setup.ps1`, `serve.ps1`, `build-koharu.ps1` and their helpers |
| `config\local-paths.json` | every path the scripts use, relative to the folder |
| `data\` | the sound-effect dictionary the server loads |
| `tests\` | the extension's JavaScript tests |
| `Setup.bat`, `Start BireLate.bat` | the same launchers as in the win64 zip |

`scripts\build-koharu.ps1` creates a directory junction
`koharu\crates\birelate-server` pointing at `server\`, so the workspace (whose
members are `crates/*`) includes the server, then runs cargo inside `koharu\`.
`server\` stays the only copy of the server's source; `.gitignore` keeps the
junction out of git.

## Requirements

- **Windows 10 or 11, x64.** No CUDA toolkit is needed to build: the CUDA
  libraries are downloaded when the server first runs.
- **Rust 1.95 or newer** with the `x86_64-pc-windows-msvc` toolchain, installed
  with rustup (one dependency, `sysinfo` 0.39.6, declares 1.95 as its minimum).
  Tested with 1.97.1.
- **Visual Studio 2022 Build Tools** (or Visual Studio 2022) with the **Desktop
  development with C++** workload and a Windows 10/11 SDK. Tested with Windows SDK
  10.0.26100.
- **LLVM**: `clang-cl` and `libclang`. Tested with 22.1.8.
- **CMake**. Tested with the copy that comes with the VS 2022 Build Tools (3.31.6).
- **Ninja**. Tested with 1.13.2.
- **Node.js**, only for the JavaScript tests. Tested with 24.18.0.

`build-koharu.ps1` adds these locations to `PATH` when they exist; anything
installed elsewhere must already be on `PATH`:

| Tool | Where the script looks |
|---|---|
| cargo | `%USERPROFILE%\.cargo\bin` (rustup's default) |
| LLVM | `C:\Program Files\LLVM\bin`. Only there does the script also set `LIBCLANG_PATH`; with LLVM elsewhere, set `LIBCLANG_PATH` to its `bin` folder yourself. |
| Ninja | the winget package folder (`winget install Ninja-build.Ninja`) and `%LOCALAPPDATA%\Microsoft\WinGet\Links` |
| CMake | `C:\Program Files\CMake\bin`, then `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` |

The script prints what it found for `cargo`, `cmake`, `ninja`, `clang-cl` and
`LIBCLANG` before building; `NOT FOUND` means that tool is missing.

## Use a short path

Clone or extract the source to a short path, for example `C:\bl` (so that
`C:\bl\koharu\Cargo.toml` exists); everything below calls it "the folder".
GitHub's source zip holds a single top-level folder: move or rename **that
folder**. MSVC and CMake still enforce Windows' 260-character path limit, and
cargo builds the C/C++ dependencies under deep
`target\<profile>\build\<crate>-<hash>\out\...` folders inside
`koharu\target`, so a long path fails with errors such as
`MSB3191`. The build has only been verified with a short target folder (24
characters); `C:\bl\koharu\target` is shorter than that.

## Build

From the folder, in a Command Prompt (not a PowerShell 7 window: Windows
PowerShell started from PowerShell 7 inherits its module path and loses some
commands):

```
powershell -ExecutionPolicy Bypass -File scripts\build-koharu.ps1 build --release -p birelate-server
```

Run it **bare, not piped** into anything: Windows PowerShell 5.1 can report a
bogus exit code 255 for a piped build that succeeded. The script's last lines
say `--- cargo exit: N` and where the output went. All arguments are passed to
cargo unchanged.

**Network at build time:** crates.io for the Rust dependencies, and
download.pytorch.org: the `koharu-torch-sys` build script downloads the CPU
libtorch wheel (about 123 MB) into `%LOCALAPPDATA%\koharu\packages` and compiles
`koharu-torch.dll` against it. Later builds reuse it; delete that folder when
you stop building. Cargo keeps its own download cache in `%USERPROFILE%\.cargo`.

**Time and disk:** a cold release build compiles about 570 crates. One measured
build took about 5 minutes and used about 4 GB under the target folder; with the
debug builds from the tests as well, the target folder reached about 27 GB.
Expect longer on a slower CPU.

The results are `koharu\target\release\birelate-server.exe` and
`koharu\target\release\koharu-torch.dll`. Then either:

- create `bin\` in the folder and copy both files into it (the layout of the
  win64 zip), or
- leave them where they are: when there is no `bin\birelate-server.exe`,
  `serve.ps1` runs the build from `koharu\target\release` (falling back to
  `koharu\target\debug` with a warning).

If `bin\birelate-server.exe` exists, `serve.ps1` always uses it, so copy a new
build there or delete `bin\`. Then continue with `README.md` from **Run
Setup.bat**. To load the extension (see `INSTALL-FIREFOX.txt`), choose
`extension\manifest.json` as a temporary add-on, or zip the contents of
`extension\`, with `manifest.json` at the top of the zip, and rename it to
`.xpi`; that is how the release's `.xpi` is made.

## Flags used for the shipped binaries

The two files in the win64 zip's `bin\` were built with the command above
(`build --release --offline -p birelate-server`; `--offline` only because the
crate cache had been filled beforehand), run from a wrapper script that first
set the following. `<workspace>` is the build folder, which held this source
tree as `<workspace>\share`, the crate cache as `<workspace>\cargo` and the
target folder as `<workspace>\target`.

| Setting | Value |
|---|---|
| `CARGO_HOME` | `<workspace>\cargo`, so crate sources are extracted there rather than under the user profile |
| `CARGO_TARGET_DIR` | `<workspace>\target` |
| `RUSTFLAGS` | `--remap-path-prefix=<workspace>=birelate-share --remap-path-prefix=%USERPROFILE%=~ --remap-path-prefix=<workspace>\cargo=cargo --remap-path-prefix=<workspace>\share=birelate -Clink-arg=/PDBALTPATH:%_PDB%` (rustc applies the last matching remap, so the order matters; `/PDBALTPATH:%_PDB%` makes the linker record only the `.pdb` file's name, not its full path) |
| `CXXFLAGS` | `/clang:-ffile-prefix-map=%LOCALAPPDATA%\koharu\packages=koharu-packages`, for the CPU libtorch headers that `koharu-torch-sys` compiles against |
| `PATH` | `%USERPROFILE%\.cargo\bin` put first |

These settings change only the file paths embedded in the binaries (in panic
messages, assertion texts and the debug-file reference), not the code that is
generated; a build without them works the same. They do not reach the C
compilers used by `aws-lc-sys` and `libz-sys`, so the C source-file paths
compiled into `birelate-server.exe` from those crates still name the build
workspace's crate cache (`<workspace>\cargo\registry\src\...`, in both its long
and short 8.3 forms). That is a build-folder path; it contains no personal data.

## Tests

The tests run on the CPU and need no GPU and no downloaded models. From the
folder, in a Command Prompt:

```
powershell -ExecutionPolicy Bypass -File scripts\build-koharu.ps1 test -p birelate-server
powershell -ExecutionPolicy Bypass -File scripts\build-koharu.ps1 test -p koharu-scene -p koharu-pipeline -p koharu-renderer -p koharu-translator --lib
```

For the extension, run each test file on its own from the folder root:

```
node --test tests\cache-key.test.js
```

or all of them:

```
for %f in (tests\*.test.js) do node --test "%f"
```

A few tests are ignored by default, and some look for the CCWildWords
font in `%LOCALAPPDATA%\koharu\packages` and skip themselves when it is absent.
