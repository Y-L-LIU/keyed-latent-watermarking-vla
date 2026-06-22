#!/usr/bin/env bash
# Regenerate every CPU-only paper asset (tables/figures) from existing per-episode
# data. Skips the two GPU steps (latent_bandstop, recovery) -- run those separately.
set -uo pipefail
HERE="$(dirname "$0")"
for s in sec5.1_spectrum sec5.1_bandstop sec6.3_verification sec6.5_identification \
         sec7.1_uniqueness sec7.2_unforgeability; do
  echo "=== $s ==="; bash "$HERE/$s.sh" || echo "!! $s failed"
done
