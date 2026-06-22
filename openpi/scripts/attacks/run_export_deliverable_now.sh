#!/bin/bash
# Export per-episode raw scores for every config that is re-derivable from
# EXISTING rollouts (no GPU needed): pi0.5 libero + robotwin, partial+map,
# the 4 canonical conditions (clean + clip_1.0 + smooth/ema_0.5 + jitter_0.01).
set -e
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:src
PY=.venv/bin/python
EXPORT=scripts/attacks/export_per_episode_scores.py
OUT=/workspace/vla/attack_c_data/per_episode_scores
mkdir -p "$OUT"

# tag -> schema attack name
declare -A ATTACK_NAME=( [none]=clean [clip_1.0]=clip [smooth_0.5]=ema [jitter_0.01]=jitter )

# ---- libero_10, pi0.5, partial+map, D_env/D_raw = 7/32 = 0.21875 ----
LIBERO_ROOT=/workspace/scratch/anon/libero10_wm_postprocess_full
for tag in none clip_1.0 smooth_0.5 jitter_0.01; do
  a=${ATTACK_NAME[$tag]}
  echo "[$(date +%H:%M:%S)] libero $tag -> $a"
  $PY $EXPORT --rollout-dir "$LIBERO_ROOT/$tag/$tag/task_rollout" \
    --out "$OUT/pi05_libero10_partial_map_${a}.csv" \
    --model pi0.5 --dataset libero_10 --attack "$a" \
    --obs partial --obs-ratio 0.21875 --recovery map --null-count 32
done

# ---- robotwin (aloha), pi0.5, partial+map, D_env/D_raw = 14/32 = 0.4375 ----
RT_ROOT=/workspace/scratch/anon/robotwin2/wm_eval
for tag in none clip_1.0 smooth_0.5 jitter_0.01; do
  a=${ATTACK_NAME[$tag]}
  echo "[$(date +%H:%M:%S)] robotwin $tag -> $a"
  $PY $EXPORT --rollout-dir "$RT_ROOT/$tag" \
    --out "$OUT/pi05_robotwin_partial_map_${a}.csv" \
    --model pi0.5 --dataset robotwin10 --attack "$a" \
    --obs partial --obs-ratio 0.4375 --recovery map --null-count 32
done

echo "[$(date +%H:%M:%S)] DONE. CSVs in $OUT"
ls -la "$OUT"
