#!/usr/bin/env bash
# Run a campaign launcher that carries hardcoded /workspace/vla paths + .venv python,
# rewriting them to the local environment (/workspace/vla + /usr/bin/python3.11) on the fly.
#   usage: _remap_run.sh /workspace/vla/attack_c_data/campaign/scripts/<launcher>.sh [args]
set -euo pipefail; source "$(dirname "$0")/../env.sh"
remap_and_run "$@"
