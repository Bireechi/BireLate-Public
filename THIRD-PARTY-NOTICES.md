# Third-party notices

This file lists what BireLate contains, what it is built from, and what it
downloads on your machine, with the licence of each. It is a list of notices, not
legal advice. Where a licence is given "as labelled", it is the label on the
publisher's page; read the licence itself before relying on it.

Model weights and GPU runtimes are **not** included in this distribution. They are
downloaded on your machine, either by `scripts\setup.ps1` or by the server the first
time it needs them (sections 4 and 5).

---

## 1. BireLate

BireLate (the Firefox extension, `birelate-server`, the scripts and the data files)
is licensed under the GNU General Public License, version 3 (`GPL-3.0-only`).
BireLate's source is its GitHub repository ("this repository" below). Each
release offers two downloads made from it: `BireLate-<version>-win64.zip` ("the
win64 zip"), the server and scripts with the two binaries, and
`BireLate-extension-<version>.xpi`, the extension, which contains its complete
source. The full licence text is in `LICENSE`, in this repository and in the
win64 zip.

The win64 zip's `bin\birelate-server.exe` and `bin\koharu-torch.dll` are built
from the source in this repository, at the tag of their release. This repository
contains all of BireLate's source and the modified Koharu's (section 2), apart
from a few upstream files that are not build inputs (listed at the end of
section 2). The third-party Rust crates (section 3) are fetched from crates.io
at build time, and so is the CPU libtorch wheel that `koharu-torch.dll` is
compiled against (from download.pytorch.org).

---

## 2. Koharu (modified)

This distribution contains a **modified version of Koharu**, a manga translation
toolkit by the Koharu authors:

- Upstream: https://github.com/mayocream/koharu (now redirects to
  https://github.com/koharu-rs/koharu)
- Base commit: `583ae5238263ee6c5e619a66224f0e5d65f32451`
- Licence: GPL-3.0-only (`koharu\LICENSE` in this repository)

Later upstream versions of Koharu are offered under different licence terms. This
distribution is based on the GPL-3.0 code at the commit above and is distributed
under GPL-3.0.

### Modification notice (GPL-3.0 section 5(a))

**This copy of Koharu was modified by the BireLate contributors in 2026.** The
modified source is the `koharu\` directory of this repository; comparing it with
upstream commit `583ae52` shows every change. In summary: the pipeline stages were
extended with OCR engine routing (including HunyuanOCR through a local sidecar and
an Ollama vision engine) and re-reads of rotated, clipped and cut text; text
detection and erase-mask construction were reworked, including sound-effect
handling; the renderer's lettering, layout and fitting were changed; the translator
gained new prompting (context from earlier pages, glossary entries, sound-effect
handling), response repair
and truncation detection, constrained-sampling changes in the llama.cpp bindings,
and an Ollama provider over the OpenAI-compatible API; model and font-catalog
downloads were pinned to exact Hugging Face revisions; model residency gained
unload and reload support, VRAM accounting and release of cached GPU memory; and
progress reporting was added. The desktop-application crates are excluded from the
workspace build, and `birelate-server` is the default build target.

### Sub-crates under MIT OR Apache-2.0

These Koharu crates declare `MIT OR Apache-2.0` in their `Cargo.toml` and keep that
licence: `koharu-bindgen`, `koharu-diffusion`, `koharu-diffusion-sys`,
`koharu-llama`, `koharu-llama-sys`, `koharu-runtime`, `koharu-torch` and
`koharu-torch-sys`. Four of them (`koharu-llama`, `koharu-runtime`, `koharu-torch`,
`koharu-torch-sys`) carry BireLate modifications, covered by the notice above. The
MIT and Apache-2.0 licence texts are reproduced in `THIRD-PARTY-RUST.txt`, which is
in this repository and in the win64 zip; this repository also has them as
`koharu\crates\koharu-torch\LICENSE-MIT` and `LICENSE-APACHE`.

`koharu-torch` and `koharu-torch-sys` originate from tch-rs, the Rust bindings for
libtorch by the tch-rs contributors (https://github.com/LaurentMazare/tch-rs),
licensed MIT OR Apache-2.0.
`bin\koharu-torch.dll` is compiled from the C++ shim in `koharu-torch-sys\libtch`,
which includes the stb_image, stb_image_write and stb_image_resize headers by Sean
Barrett (public domain or MIT, at your choice; the licence text is at the end of
each header). The DLL loads libtorch at run time; libtorch itself is not included
(section 4), but the DLL does contain code from the libtorch C++ headers (see
"PyTorch licence" below).

This repository also contains vendored C headers, used to generate the Rust
bindings to libraries that are downloaded at run time (section 4):

- `koharu\crates\koharu-llama-sys\include\` (`llama.h`, `mtmd.h`,
  `mtmd-helper.h`, `ggml*.h`, `gguf.h`), from llama.cpp and ggml
  (https://github.com/ggml-org/llama.cpp): MIT, Copyright (c) 2023-2026 The
  ggml authors.
- `koharu\crates\koharu-diffusion-sys\include\stable-diffusion.h`, from
  stable-diffusion.cpp (https://github.com/leejet/stable-diffusion.cpp): MIT,
  Copyright (c) 2023 leejet.

The MIT licence text is in `THIRD-PARTY-RUST.txt`.

### PyTorch licence

`bin\koharu-torch.dll` is compiled against the libtorch C++ headers, and the
inline code from those headers (for example from `ATen`, `c10` and
`torch\csrc\jit`) is compiled into the DLL, so PyTorch's licence applies to that
part of it. The text below is PyTorch's `LICENSE` at tag v2.12.1
(https://github.com/pytorch/pytorch/blob/v2.12.1/LICENSE), reproduced in full:

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### Third-party material excluded from this repository

A few files of upstream Koharu at commit `583ae52` are left out of this
repository and never published with BireLate: its website documentation (`koharu\docs`), the benchmark fixtures under
`koharu\crates\koharu-ml\benches\fixtures`, a comic-download script
(`koharu\scripts\download_bluearchive_comics.ts`) and stale Python bytecode
(`koharu\scripts\__pycache__`). The documentation's screenshots and some of the
fixture images show copyrighted comic artwork, the script downloads such
artwork, and the bytecode is compiled from scripts that upstream had already
deleted, so it is object code without its source. None of them is a build
input (the fixtures are read only by koharu-ml's benchmarks and nine of its unit
tests, seven of them ignored by default; none is among the tests BUILDING.md
runs, and the two that run by default fail in this repository because the
image is left out); they remain in upstream's
repository at that commit.

---

## 3. Rust crate dependencies

The third-party Rust crates linked into `birelate-server.exe`, with their licence
expressions and licence texts, are listed in `THIRD-PARTY-RUST.txt`.

---

## 4. Runtime components downloaded on your machine (not included)

| Component | Version | Source | Licence | Downloaded by |
|---|---|---|---|---|
| NVIDIA CUDA libraries (PyPI wheels): `nvidia-cuda-runtime` 13.0.48, `nvidia-cublas` 13.0.0.19, `nvidia-cufft` 12.0.0.15, `nvidia-curand` 10.4.0.35, `nvidia-cuda-nvrtc` 13.0.88, `nvidia-cuda-cupti` 13.0.48, `nvidia-nvjitlink` 13.0.39, `nvidia-cusparse` 12.6.2.49, `nvidia-cusolver` 12.0.3.29 | as listed | https://pypi.org (files.pythonhosted.org) | NVIDIA CUDA Toolkit End User License Agreement (see each package's PyPI page) | the server, at its first start |
| NVIDIA cuDNN: `nvidia-cudnn-cu13` | 9.20.0.48 | https://pypi.org | NVIDIA cuDNN Software License Agreement | the server, at its first start |
| PyTorch (libtorch), Windows CUDA 13.0 wheel | 2.12.1+cu130 | https://download.pytorch.org/whl/cu130/ | BSD-3-Clause style (text in section 2, "PyTorch licence"). The wheel also contains `libiomp5md.dll` and `libiompstubs5md.dll` (Intel OpenMP), `uv.dll` (libuv) and `zlibwapi.dll` (zlib), which are extracted with libtorch; they come with the downloaded wheel under their own licences. | the server, at its first start |
| llama.cpp (Koharu's build) | b9982 | https://github.com/mayocream/koharu/releases/tag/llama.cpp-b9982 | MIT. `llama.dll` also contains the NVIDIA CUDA runtime, linked in statically (it imports only `cublas64_13.dll`), which is under the NVIDIA CUDA Toolkit End User License Agreement. | the server, at its first start |
| stable-diffusion.cpp (Koharu's build) | master-769-cc73429 | https://github.com/mayocream/koharu/releases/tag/stable-diffusion.cpp-master-769-cc73429 | MIT. `stable-diffusion.dll` also contains the NVIDIA CUDA runtime, linked in statically, under the NVIDIA CUDA Toolkit End User License Agreement. | the server, at its first start |
| `cudart64_13.dll`, taken from the `nvidia-cuda-runtime` wheel | 13.0.48 | https://pypi.org/project/nvidia-cuda-runtime/ | NVIDIA CUDA Toolkit End User License Agreement | `setup.ps1`, always |
| llama.cpp, `llama-b10502-bin-win-cuda-13.3-x64.zip` | b10502 | https://github.com/ggml-org/llama.cpp/releases/tag/b10502 | MIT. The zip also contains `libomp140.x86_64.dll`, the LLVM OpenMP runtime, under its own licence. | `setup.ps1`, only if you accept the HunyuanOCR licence |
| NVIDIA CUDA 13.3 runtime libraries packaged by llama.cpp, `cudart-llama-bin-win-cuda-13.3-x64.zip` | b10502 | https://github.com/ggml-org/llama.cpp/releases/tag/b10502 | NVIDIA CUDA Toolkit End User License Agreement | `setup.ps1`, only if you accept the HunyuanOCR licence |
| Python, Windows embeddable package | 3.12.10 | https://www.python.org/ftp/python/3.12.10/ | Python Software Foundation License. The package also contains libraries such as OpenSSL, libffi and SQLite, and Microsoft's `vcruntime140.dll`, under their own licences. | `setup.ps1`, only if you accept the HunyuanOCR licence |
| Pillow | 12.3.0 | https://pypi.org/project/pillow/ | MIT-CMU (HPND). The wheel's compiled modules bundle libjpeg-turbo, libwebp, FreeType, HarfBuzz, libavif and other libraries under their own licences, collected in the wheel's `pillow-12.3.0.dist-info\licenses\LICENSE`. | `setup.ps1`, only if you accept the HunyuanOCR licence |

The server's downloads go to `local\runtime\koharu-packages`. `setup.ps1`'s go to
`local\runtime\cuda-bootstrap`, `local\runtime\llama-ocr` and `local\runtime\python`.
Every file `setup.ps1` downloads is checked against a pinned size and SHA256.

The server rows above are what a machine with a CUDA 13 NVIDIA GPU, the only
configuration BireLate supports, downloads. On a machine without one, Koharu's
runtime code would select other builds of the same projects instead: the AMD
ROCm (HIP) or Vulkan builds of llama.cpp and stable-diffusion.cpp from the same
Koharu releases, and the CPU torch wheel from download.pytorch.org or AMD's
ROCm torch wheels from repo.amd.com. Those are untested with BireLate and not
listed here; each comes under its publisher's licence.

---

## 5. Models downloaded on your machine (not included)

All model downloads are pinned to the exact revision shown.

### Used by the default configuration

| Model | Repository and revision | Licence | Downloaded | Notes |
|---|---|---|---|---|
| RF-DETR text and bubble detector | `mayocream/koharu-layout-rfdetr-seg-2xl-1152` @ `aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2` | "other", as labelled | by the server, on the first translation | The model card gives Apache-2.0 for the code. The model is fine-tuned on Manga109, whose images are for academic and non-commercial use, and the card leaves compliance with that to the user. |
| LaMa manga inpainter | `mayocream/lama-manga` @ `f91c85b26913b3e83f9877867b4c336da3675238` | MIT | by the server, on the first translation | |
| Gemma 4 26B A4B translator, file `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` @ `7b92b5b28818151e8669af2e45e88d6086f490dd` | Apache-2.0 (https://ai.google.dev/gemma/docs/gemma_4_license) | by the server, on the first translation | Google's Gemma prohibited use policy is linked from the licence page: https://ai.google.dev/gemma/prohibited_use_policy |
| PaddleOCR-VL 1.6 | `PaddlePaddle/PaddleOCR-VL-1.6` @ `66317acc4c9fc17bd154591ce650735cd2855f3e` | Apache-2.0 | by the server, when first needed | The OCR engine when HunyuanOCR is not installed; also used for some Korean re-reads and some Japanese paged pages. |
| manga-ocr | `mayocream/manga-ocr` @ `4380edba990b959c508752350955350c1c80c31c` | Apache-2.0 | by the server, when first needed | Used for Japanese paged manga (the extension selects it automatically). Trained on Manga109-s. |
| HunyuanOCR 1.5, BF16 GGUF model and mmproj | `prithivMLmods/HunyuanOCR-1.5-GGUF-Updated` @ `9ddd3b47beb0de305ecd89a717748bac080d7aee` | Tencent Hunyuan Community License Agreement | by `setup.ps1`, only if you accept that licence | The repository labels itself apache-2.0, but as a conversion of HunyuanOCR it remains under Tencent's licence. See section 6. |
| HunyuanOCR chat template, `chat_template.jinja` | `tencent/HunyuanOCR` @ `449e7d471a8a1ef5bd5d652e4881183d7252cbc7` | Tencent Hunyuan Community License Agreement | by `setup.ps1`, only if you accept that licence | See section 6. |
| The agreement's text, `LICENSE` | `tencent/HunyuanOCR` @ `449e7d471a8a1ef5bd5d652e4881183d7252cbc7` | (the agreement itself) | by `setup.ps1`, only if you accept that licence | Saved as `local\models\hunyuan-gguf\LICENSE`, next to the model. |
| Font catalog (`index.json`) and the font files actually used | dataset `mayocream/fonts` @ `304901a868bebd5f31f95664a462e5839873d939` | none stated on the dataset | by the server, on the first translation | The default lettering font is CCWildWords; the font file's embedded trademark notice reads "CCWildWords is a trademark of Comic Book Fonts LLC, 2023. All rights reserved." Its licence terms are unknown. The catalog lists 640 font families, and its own index marks the licensing of 413 of them as `unknown-review-required`. No font is bundled; only the files needed are fetched. Starting the server with `"Start BireLate.bat" -FontFamily Arial` (server flag `--font-family Arial`) letters with a system font instead. |

### Downloaded only if you select another engine

| Engine | Repository and revision | Licence, as labelled |
|---|---|---|
| Baberu OCR (popup) | `genshiai-daichi/baberu-ocr` @ `d9cc13153e9a1cd8fdfa3b7b1cc329da2020aeae`; `facebook/dinov2-base` @ `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | Apache-2.0; Apache-2.0 |
| AOT inpainter (popup) | `mayocream/aot-inpainting` @ `cffe2346ac2b5ebe1f2d61335d602d12cc144c6f` | MIT |
| FLUX.2 Klein inpainter (popup) | `unsloth/FLUX.2-klein-4B-GGUF` @ `0084d1df98e2e2137fe776d55170bc4792ec1d66`; `black-forest-labs/FLUX.2-small-decoder` @ `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`; `unsloth/Qwen3-4B-GGUF` @ `22c9fc8a8c7700b76a1789366280a6a5a1ad1120` | Apache-2.0 each |
| RORem mixed inpainter (popup) | `mayocream/RORem-mixed-GGUF` @ `62c75b3e6f078a19e2698b0f677e8a4aa4c9ea56`; `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` @ `115134f363124c53c7d878647567d04daf26e41e` | CreativeML Open RAIL++-M (`openrail++`), which carries use restrictions |
| `--refine-text-mask` (server flag, off by default) | `mayocream/manga-text-segmentation-2025` @ `efd866e3ac6595ea20722f35ae343c403056ba76` | none stated |

Choosing another translator with `-Llm` (server flag `--llm`) downloads that model
from its own repository, under its own licence; apart from `gemma4-31b-it`, those
downloads are not pinned and take whatever revision is current. The Ollama options (OCR engine and translation
provider) use models already in your own Ollama installation; BireLate downloads
nothing for them.

---

## 6. HunyuanOCR and the Tencent Hunyuan Community License Agreement

HunyuanOCR is made by Tencent and is licensed under the **Tencent Hunyuan Community
License Agreement**. The GGUF model files BireLate uses are a community conversion
of it and are covered by the same agreement.

**BireLate does not redistribute HunyuanOCR.** `scripts\setup.ps1` shows the licence
terms and downloads the files only after you type `YES` (or pass
`-AcceptHunyuanLicense`). If you do not accept, nothing is downloaded and the server
uses PaddleOCR-VL instead (`--hunyuan-substitute paddleocr-vl-1.6`). Run
`Setup.bat -SkipHunyuan` to skip the question.

**Territory.** The agreement does not apply in, and grants no rights in, the
**European Union, the United Kingdom or South Korea**. It also prohibits using,
reproducing, modifying, distributing or displaying the model, its output or its
results outside its territory. If you are in one of those places, do not accept it;
BireLate works without HunyuanOCR.

**The agreement.** A copy of the agreement, at the revision BireLate pins:
https://huggingface.co/tencent/HunyuanOCR/blob/449e7d471a8a1ef5bd5d652e4881183d7252cbc7/LICENSE.
After you accept, `setup.ps1` also saves that copy next to the model, as
`local\models\hunyuan-gguf\LICENSE`.

**Notice** (the text the agreement requires):

> Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.

**Provider.** BireLate is provided by the BireLate contributors. Tencent is not
affiliated with, associated with, sponsoring or endorsing BireLate.

**Use restrictions.** Use of HunyuanOCR is subject to the use restrictions in
sections 5(a) and 5(b) of the agreement: it must comply with applicable laws and
with Tencent's Acceptable Use Policy (Exhibit A of the agreement, at the link
above), and neither the model nor its output may be used to improve any other AI
model.

---

## 7. `data\sfx-dictionary.json`

BireLate's own table of common onomatopoeia (sound effects) and their usual English
renderings, applied after the translator: it replaces the translation of a matching
sound effect. It is part of BireLate and covered by
`LICENSE` (GPL-3.0).
