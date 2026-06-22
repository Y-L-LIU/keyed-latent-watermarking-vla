#!/usr/bin/env bash
# Shared environment for all experiment wrappers.
# These wrappers DO NOT copy any code -- they only `exec` the original scripts
# in place, after setting the paths/env each one needs. Source this from a wrapper:
#     source "$(dirname "$0")/../env.sh"
#
# Single source of truth for "where things live" on THIS node. The original
# campaign launchers were authored on a different node under /workspace/vla
# (with a project-local .venv); on this node that remaps to /workspace/vla and the
# system python /usr/bin/python3.11. Wrappers that touch those launchers apply
# the remap on the fly (see rollout/_remap_run.sh).

export VLA=/workspace/vla
export PY=/usr/bin/python3.11

# proxy (PyPI + HF only reachable through it; direct is blocked)

# openpi / JAX
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$VLA/openpi-cache}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$VLA/jax_cache}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

# headless rendering: this node has no EGL/Vulkan ICD -> CPU rendering only
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

# /workspace/vla -> /workspace/vla remap helper for the campaign launchers
# usage: remap_and_run /path/to/orig.sh [args...]
remap_and_run() {
  local orig="$1"; shift
  local tmp; tmp="$(mktemp /tmp/exp_remap_XXXX.sh)"
  sed -e 's#/workspace/vla#/workspace/vla#g' \
      -e 's#/workspace/vla/\.venv/bin/python#/usr/bin/python3.11#g' \
      -e 's#\.venv/bin/python#/usr/bin/python3.11#g' \
      "$orig" > "$tmp"
  echo "[exp] remapped $orig -> $tmp" >&2
  exec bash "$tmp" "$@"
}
