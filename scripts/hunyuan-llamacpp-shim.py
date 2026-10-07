#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""The sidecar's protocol, served by llama.cpp: a shim so NOTHING else changes.

    (started by serve.ps1, which also starts the llama-server it fronts;
     not run bare -- it needs that llama-server behind it)

WHY IT EXISTS. The spot call's ~2.6 s is decode length at the HF (Python
transformers) sidecar's ~40-60 tokens/s, the model ignores prompt-side output
trimming (20/20 boxes-only responses still transcribe), and the remaining
levers -- an engine-class decode rate and a GRAMMAR that structurally forbids
the `text` key -- both live in llama.cpp. This shim speaks the HF sidecar's
exact wire protocol on its port (11436), so `ocr.rs` and `serve.ps1` (which
ADOPTS an existing 11436 listener) run against llama-server unchanged.

THE CONTRACT it replicates, byte-for-byte where it matters:
  GET  /api/tags -> {"models": [{"name": "hunyuan-ocr-1.5"}]}
  POST /api/chat -> {"model", "message": {"role", "content"},
                     "score": null | {"mean_logprob", "prob", "tokens"}, "done"}
  options.rotate_ccw: the image is rotated BEFORE reading with the EXACT PIL
  call the sidecar uses (`rotate(angle, expand=True, fillcolor=white,
  bicubic)`) -- the flip refine's margins were measured through that call.

THE SCORE CAVEAT, stated rather than hidden: mean_logprob here is the mean of
llama-server's returned per-token logprobs (its final sampling distribution,
`repeat_penalty` 1.08 applied to mirror the sidecar's chain). It is the same
QUANTITY as the sidecar's score, not the same NUMBER: kernels and the
distribution being scored differ. Whether the shipping margins (0.17
orientation, 0.17 flip) still separate on it is a question for knife-edge
reads, not for latency numbers.

--boxes-grammar attaches a GBNF that forbids the `text` key outright -- the
output-side lever no prompt could reach. OFF by default.
"""
import argparse
import base64
import ctypes
import io
import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIRE_NAME = "hunyuan-ocr-1.5"

# Forbids "text" by construction: an element is {"box": [n, n, n, n]} and
# nothing else. Whitespace is left flexible so the grammar follows the model's
# natural spacing instead of fighting the distribution token by token.
BOXES_GRAMMAR = r'''
root ::= ws "[" ws (elem (ws "," ws elem)*)? ws "]" ws
elem ::= "{" ws "\"box\"" ws ":" ws "[" ws int ws "," ws int ws "," ws int ws "," ws int ws "]" ws "}"
int ::= [0-9] [0-9]? [0-9]? [0-9]?
ws ::= [ \t\n]*
'''


def make_handler(llama_url, grammar, num_predict_cap):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # quiet: no per-request log
            pass

        def do_GET(self):
            if self.path == "/api/tags":
                # BACKEND HEALTH RIDES THE PROBE -- the no-orphans rule. A shim
                # whose llama-server has died answers every read with a 500
                # URLError, and serve.ps1's load probe takes a good answer
                # here as "sidecar up, leaving it alone" -- a squatter that
                # would silently 500 a whole chapter. A dead backend is a 503,
                # so the probe starts a fresh stack instead of adopting a corpse.
                try:
                    urllib.request.urlopen(llama_url + "/health", timeout=1.5).read()
                except Exception as e:
                    return self._json(503, {"error": f"llama backend down: {type(e).__name__}"})
                # The capabilities list is LOAD-BEARING: ocr.rs sets
                # rotate_capable from it, and without "rotate_ccw" the engine
                # disables the upright pass, the spot rescue AND the flip
                # refine wholesale -- silently. The shim
                # really does implement rotate_ccw (the sidecar's exact PIL
                # call), and "score" rides the logprobs mapping.
                return self._json(200, {"models": [{"name": WIRE_NAME}],
                                        "capabilities": ["score", "rotate_ccw"]})
            return self._json(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                if self.path != "/api/chat":
                    return self._json(404, {"error": "unknown path"})
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                msg = req["messages"][-1]
                prompt = msg.get("content") or ""
                images = msg.get("images") or []
                if not images:
                    return self._json(400, {"error": "no image in request"})
                png = base64.b64decode(images[0])
                opts = req.get("options") or {}
                max_new = min(int(opts.get("num_predict") or 128), num_predict_cap)
                rotate_ccw = opts.get("rotate_ccw")
                if rotate_ccw is not None and float(rotate_ccw) != 0.0:
                    from PIL import Image
                    im = Image.open(io.BytesIO(png)).convert("RGB")
                    # Byte-exact the sidecar's call: the flip refine's margins
                    # were measured through it.
                    im = im.rotate(float(rotate_ccw), expand=True,
                                   fillcolor=(255, 255, 255), resample=Image.BICUBIC)
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    png = buf.getvalue()
                body = {
                    "model": "hunyuan-ocr",
                    "messages": [{"role": "user", "content": [
                        # IMAGE FIRST, prompt second -- the sidecar's exact
                        # order ({"type":"image"},{"type":"text"}), and the
                        # order the model's task selection is trained on.
                        # Swapped, the model answers with the right TEXT in the
                        # wrong FORMAT on most pages (plain lines, no boxes)
                        # even with the template in force (checked via /props).
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," +
                                   base64.b64encode(png).decode()}},
                        {"type": "text", "text": prompt},
                    ]}],
                    "max_tokens": max_new,
                    "temperature": 0.0,
                    "repeat_penalty": 1.08,
                    "logprobs": True,
                }
                # The grammar constrains SPOT calls only, recognised by the
                # task phrase the spot prompt opens with. Ordinary reads
                # ("请识别图中的所有文字...") must stay unconstrained -- forcing
                # box-JSON onto a free-text read would replace every
                # transcription in a chapter run with junk geometry.
                if grammar and prompt.startswith("检测并识别"):
                    body["grammar"] = BOXES_GRAMMAR
                up = urllib.request.Request(
                    llama_url.rstrip("/") + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(up, timeout=600) as r:
                    resp = json.loads(r.read())
                choice = resp["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                score = None
                content = ((choice.get("logprobs") or {}).get("content")) or []
                if content:
                    lps = [t["logprob"] for t in content if "logprob" in t]
                    if lps:
                        mean = sum(lps) / len(lps)
                        score = {"mean_logprob": mean, "prob": math.exp(mean),
                                 "tokens": len(lps)}
                return self._json(200, {
                    "model": req.get("model", WIRE_NAME),
                    "message": {"role": "assistant", "content": text.strip()},
                    "score": score,
                    "done": True,
                })
            except Exception as e:  # one bad crop must not kill the shim
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def _pid_alive(pid):
    """Windows liveness by handle + exit code; os.kill(pid, 0) is not reliable here."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def _watchdog(llama_url, watch_pid, llama_pid):
    """Self-terminate rather than squat -- the no-orphans rule.

    Two exits, both observed failure modes rather than hypotheticals:
    - PARENT GONE: serve.ps1 (or whatever launched us) died without teardown --
      the path that can leave a shim squatting port 11436 with a dead
      backend, silently 500ing a whole chapter render. Nothing will ever stop
      this process now, so it stops itself, taking its llama-server child
      along when it was given the pid.
    - BACKEND DEAD: four consecutive failed /health probes (~40 s) mean every
      future /api/chat is a 500 URLError; exiting frees the port so the next
      serve.ps1 can bind a healthy stack.
    `os._exit`, not `sys.exit`: serve_forever's threads must not keep the
    process alive through a clean-shutdown attempt.
    """
    failures = 0
    while True:
        time.sleep(10)
        if watch_pid and not _pid_alive(watch_pid):
            if llama_pid and _pid_alive(llama_pid):
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x0001, False, llama_pid)  # PROCESS_TERMINATE
                if handle:
                    kernel32.TerminateProcess(handle, 1)
                    kernel32.CloseHandle(handle)
            print("shim: parent gone -- exiting (and stopping llama-server)", flush=True)
            os._exit(2)
        try:
            urllib.request.urlopen(llama_url + "/health", timeout=3).read()
            failures = 0
        except Exception:
            failures += 1
            if failures >= 4:
                print("shim: llama backend dead for ~40s -- exiting rather than squat", flush=True)
                os._exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11436)
    ap.add_argument("--llama-url", default="http://127.0.0.1:11437")
    ap.add_argument("--boxes-grammar", action="store_true",
                    help="attach the GBNF that forbids the text key (spot-call "
                         "output lever; its own bench arm, never a silent default)")
    ap.add_argument("--num-predict-cap", type=int, default=4096,
                    help="matches the sidecar's post-fix ceiling")
    ap.add_argument("--watch-pid", type=int, default=0,
                    help="exit when this process (the launching serve.ps1) is "
                         "gone -- the no-orphans rule; 0 disables")
    ap.add_argument("--llama-pid", type=int, default=0,
                    help="llama-server child to stop on parent death, so "
                         "neither half of the stack outlives its session")
    a = ap.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", a.port),
        make_handler(a.llama_url, a.boxes_grammar, a.num_predict_cap))
    threading.Thread(
        target=_watchdog, args=(a.llama_url, a.watch_pid, a.llama_pid),
        daemon=True).start()
    print(f"shim: {WIRE_NAME} protocol on 127.0.0.1:{a.port} -> {a.llama_url}"
          f"{' [boxes-grammar]' if a.boxes_grammar else ''}"
          f" watch_pid={a.watch_pid or 'off'} llama_pid={a.llama_pid or 'off'}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
