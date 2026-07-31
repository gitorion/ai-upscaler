#!/usr/bin/env python3
"""
Bug-fix patch for FlashVSR-Pro's colour correction.

THE BUG
-------
With --tile-dit, the tiled pipeline runs colour correction with method='adain'. That path calls
_calc_mean_std(), which uses `.view()` on a tensor that an earlier `permute()` left non-contiguous:

    var  = feat.view(N, C, -1).var(dim=2, ...)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

PyTorch then raises:

    view size is not compatible with input tensor's size and stride
    (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

The call site swallows it (`except Exception: print("[ColorFix Error]")`), so the run continues with
NO colour correction applied — silently. That matters when tiling: each tile can drift in colour
independently, which is a direct cause of visible tile seams.

THE FIX
-------
`.reshape()` is exactly `.view()` but falls back to a copy when the tensor isn't contiguous, so it
is a safe drop-in with identical semantics. We rewrite only the `.view(` calls inside
_calc_mean_std, in whichever pipeline files define it.

Fail-soft, like patch_hq_output.py: if the expected code isn't found, nothing is modified and we
exit non-zero so setup.sh can warn. Each file is byte-compiled after patching and reverted if it
fails to parse.

Usage:  python3 patch_colorfix.py /path/to/FlashVSR-Pro
"""
import os
import py_compile
import shutil
import sys

TARGETS = [
    "diffsynth/pipelines/flashvsr_full.py",
    "diffsynth/pipelines/flashvsr_tiny.py",
    "diffsynth/pipelines/flashvsr_tiny_long.py",
]

REPLACEMENTS = [
    ("var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps",
     "var = feat.reshape(N, C, -1).var(dim=2, unbiased=False) + eps"),
    ("std = var.sqrt().view(N, C, 1, 1)",
     "std = var.sqrt().reshape(N, C, 1, 1)"),
    ("mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)",
     "mean = feat.reshape(N, C, -1).mean(dim=2).reshape(N, C, 1, 1)"),
]


def patch_file(path: str) -> str:
    """Returns 'patched', 'already', 'nomatch', or 'missing'."""
    if not os.path.isfile(path):
        return "missing"

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if all(new in src for _, new in REPLACEMENTS):
        return "already"

    hits = sum(1 for old, _ in REPLACEMENTS if old in src)
    if hits == 0:
        return "nomatch"

    patched = src
    for old, new in REPLACEMENTS:
        patched = patched.replace(old, new)

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)

    try:
        py_compile.compile(path, doraise=True)
    except Exception as exc:  # noqa: BLE001
        shutil.copy2(backup, path)
        print("[colorfix] %s failed to compile (%s) — reverted." % (path, exc), file=sys.stderr)
        return "nomatch"

    return "patched"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_colorfix.py /path/to/FlashVSR-Pro", file=sys.stderr)
        return 2

    repo = sys.argv[1]
    any_ok = False
    for rel in TARGETS:
        result = patch_file(os.path.join(repo, rel))
        print("[colorfix] %-42s %s" % (rel, result))
        if result in ("patched", "already"):
            any_ok = True

    if not any_ok:
        print("[colorfix] nothing patched — upstream may have changed _calc_mean_std.",
              file=sys.stderr)
        print("[colorfix] --color-fix will keep failing silently (run still works, no colour "
              "correction).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
