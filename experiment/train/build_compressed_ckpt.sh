#!/usr/bin/env bash
# §6.4 / §12.5 MODEL SURGERY: build a pruned or quantized copy of a descendant ckpt.
# This is the "training-side" step that produces the attacked model the compression
# eval (rollout/{openpi,lingbot}_compression.sh) then scores with the base detector.
#   usage: build_compressed_ckpt.sh <openpi|lingbot> <prune|quant> <SRC> <DST> [--prune-sparsity 0.3]
#
# openpi : openpi/scripts/attacks/build_compressed_ckpt.py
#            --src-ckpt <step dir w/ params/,assets/> --dst-ckpt <dir> --attack <prune|quant>
# lingbot: lingbot-va/wan_va/attacks/build_compressed_transformer.py
#            --src-transformer <.../transformer/> --dst-transformer <.../transformer/> --attack <prune|quant>
# (int8 quant = per-output-channel symmetric fake-quant; prune = magnitude L1, default 30%)
set -euo pipefail; source "$(dirname "$0")/../env.sh"
fam="${1:?openpi|lingbot}"; atk="${2:?prune|quant}"; src="${3:?SRC}"; dst="${4:?DST}"; shift 4 || true
case "$fam" in
  openpi)  cd "$VLA/openpi"
           exec "$PY" scripts/attacks/build_compressed_ckpt.py \
             --src-ckpt "$src" --dst-ckpt "$dst" --attack "$atk" "$@" ;;
  lingbot) cd "$VLA/lingbot-va"; export PYTHONPATH="$VLA/lingbot-va:${PYTHONPATH:-}"
           exec "$PY" wan_va/attacks/build_compressed_transformer.py \
             --src-transformer "$src" --dst-transformer "$dst" --attack "$atk" "$@" ;;
  *) echo "arg1 must be openpi|lingbot"; exit 2 ;;
esac
