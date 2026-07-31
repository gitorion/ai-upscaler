#!/usr/bin/env python3
"""
Quality patch for FlashVSR-Pro's infer.py.

WHY THIS EXISTS
---------------
FlashVSR-Pro writes its result with a hardcoded encoder. Even at its maximum
`--quality 10` that is:

    8-bit H.264, yuv420p, CRF 20, preset veryfast     (or NVENC QP 20, worse)

For this pipeline that is a hard quality ceiling: the diffusion model's output
would be crushed to 8-bit CRF-20 *before* our own encoder ever sees it, and no
later step can recover it. SeedVR2 by comparison hands us a 10-bit stream we can
copy losslessly.

This patch injects an early-return into `save_video()` that, when the env var
AIUPSCALER_HQ_OUT=1 is set, writes x265 **10-bit** at a configurable CRF
(default 12 — visually transparent) instead. upscale_video.sh then stream-copies
that result, so the whole path has exactly ONE encode, at high quality.

Design notes:
  * Opt-in via env var, so an unpatched/patched repo behaves identically unless
    we ask for the HQ writer. Nothing else changes.
  * Idempotent — re-running is safe (detects its own marker).
  * Reversible — the original is kept as infer.py.orig.
  * Fail-soft by contract: if the anchor is missing (upstream refactored), this
    exits non-zero WITHOUT modifying anything, and setup.sh only warns. You then
    get FlashVSR's default 8-bit output, which still works — just lower quality.

Usage:  python3 patch_hq_output.py /path/to/FlashVSR-Pro/infer.py
"""
import os
import shutil
import sys

MARKER = "ai-upscaler HQ output patch"

ANCHOR = 'def save_video(frames, save_path, fps=30, quality=5):'

PATCH = '''
    # ── {marker} ──
    # Opt-in high-quality writer. Upstream hardcodes 8-bit H.264 CRF 20 (preset
    # veryfast), which would cap this pipeline's quality before our encoder runs.
    # When AIUPSCALER_HQ_OUT=1 we write x265 10-bit instead and return early.
    import os as _hq_os
    if _hq_os.getenv("AIUPSCALER_HQ_OUT") == "1":
        import subprocess as _hq_sp
        import numpy as _hq_np
        _hq_crf = _hq_os.getenv("AIUPSCALER_HQ_CRF", "12")
        _hq_preset = _hq_os.getenv("AIUPSCALER_HQ_PRESET", "medium")
        _hq_frames = list(frames)
        if not _hq_frames:
            raise RuntimeError("[ai-upscaler] save_video: no frames to write")
        _hq_first = _hq_np.asarray(_hq_frames[0])
        _hq_h, _hq_w = _hq_first.shape[:2]
        _hq_dir = _hq_os.path.dirname(_hq_os.path.abspath(save_path))
        if _hq_dir:
            _hq_os.makedirs(_hq_dir, exist_ok=True)
        _hq_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (_hq_w, _hq_h), "-r", str(fps), "-i", "-",
            "-c:v", "libx265", "-crf", str(_hq_crf), "-preset", _hq_preset,
            "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error",
            save_path,
        ]
        print("[ai-upscaler] HQ writer: x265 10-bit crf=%s preset=%s -> %s"
              % (_hq_crf, _hq_preset, save_path))
        _hq_proc = _hq_sp.Popen(_hq_cmd, stdin=_hq_sp.PIPE)
        try:
            for _hq_f in _hq_frames:
                _hq_arr = _hq_np.ascontiguousarray(_hq_np.asarray(_hq_f, dtype=_hq_np.uint8))
                _hq_proc.stdin.write(_hq_arr.tobytes())
        finally:
            _hq_proc.stdin.close()
        if _hq_proc.wait() != 0:
            raise RuntimeError("[ai-upscaler] HQ writer: ffmpeg failed")
        return
    # ── end {marker} ──
'''.format(marker=MARKER)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_hq_output.py /path/to/infer.py", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not os.path.isfile(path):
        print("[patch] not found: %s" % path, file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("[patch] already applied — nothing to do")
        return 0

    if ANCHOR not in src:
        print("[patch] anchor not found (upstream changed its save_video signature).",
              file=sys.stderr)
        print("[patch] Leaving infer.py UNMODIFIED. FlashVSR will write its default",
              file=sys.stderr)
        print("[patch] 8-bit H.264 CRF 20 output — usable, but lower quality.",
              file=sys.stderr)
        return 1

    # Insert immediately after the def line and its docstring line (if present).
    lines = src.split("\n")
    out = []
    inserted = False
    for i, line in enumerate(lines):
        out.append(line)
        if not inserted and line.strip() == ANCHOR:
            # Skip past a one-line docstring so we insert into the body proper.
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.strip().startswith('"""') and nxt.strip().endswith('"""') and len(nxt.strip()) > 6:
                out.append(nxt)
                lines[i + 1] = "\x00SKIP\x00"
            out.append(PATCH)
            inserted = True

    if not inserted:
        print("[patch] failed to locate insertion point", file=sys.stderr)
        return 1

    patched = "\n".join(l for l in out if l != "\x00SKIP\x00")

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("[patch] original saved as %s" % backup)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)

    # Verify it still parses — a broken infer.py is worse than an unpatched one.
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
    except Exception as exc:  # noqa: BLE001
        shutil.copy2(backup, path)
        print("[patch] patched file failed to compile (%s) — reverted." % exc, file=sys.stderr)
        return 1

    print("[patch] applied — HQ 10-bit writer available via AIUPSCALER_HQ_OUT=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
