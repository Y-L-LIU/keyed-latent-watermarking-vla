"""Launcher that patches ``torch.load`` to keep PyTorch <2.6 behavior.

LIBERO's benchmark module calls ``torch.load`` on init-state pickles that contain
``numpy.core.multiarray._reconstruct``. Starting with PyTorch 2.6 ``weights_only``
defaults to True which rejects those globals. We can't edit the submodule
cleanly from inside a worktree, so this thin wrapper:

  1. Forces ``torch.load`` to ``weights_only=False``.
  2. Forwards remaining argv to the wrapped eval script via ``runpy.run_path``.

Usage::

    PYTHONPATH=third_party/libero \
    python scripts/attacks/run_eval_with_libero_patch.py \
        scripts/eval_libero_action_inversion_postprocess_robustness.py -- \
        --controller-postprocess none ...
"""

from __future__ import annotations

import argparse
import runpy
import sys

import torch


def _patch_torch_load() -> None:
    """Force weights_only=False so LIBERO init-state pickles deserialize."""
    _original = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("script", help="Path to the eval script to run.")
    args, forward = parser.parse_known_args()

    if "--" in forward:
        forward.remove("--")

    _patch_torch_load()

    sys.argv = [args.script, *forward]
    runpy.run_path(args.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
