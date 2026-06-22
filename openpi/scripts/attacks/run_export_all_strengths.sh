#!/bin/bash
# Export per-episode raw scores for ALL attack strengths on disk (GPU-free),
# pi0.5 libero + robotwin, partial+map. Full schema incl variant + attack_strength.
set -e
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:src
PY=.venv/bin/python
EXPORT=scripts/attacks/export_per_episode_scores.py
OUT=/workspace/vla/attack_c_data/per_episode_scores
mkdir -p "$OUT"

# tag -> "attack strength"
declare -A MAP=(
  [none]="clean "
  [clip_0.5]="clip 0.5"  [clip_1.0]="clip 1.0"  [clip_2.0]="clip 2.0"
  [smooth_0.2]="ema 0.2" [smooth_0.5]="ema 0.5" [smooth_0.8]="ema 0.8"
  [jitter_0.005]="jitter 0.005" [jitter_0.01]="jitter 0.01"
  [jitter_0.05]="jitter 0.05"   [jitter_0.1]="jitter 0.1"
)

run() {  # $1=dataset $2=root $3=ratio $4=tag $5=globsuffix
  read a s <<< "${MAP[$4]}"
  local sfx; [ -n "$s" ] && sfx="_${s}" || sfx=""
  echo "[$(date +%H:%M:%S)] $1 $4 -> attack=$a strength=${s:-NA}"
  $PY $EXPORT --rollout-dir "$2" \
    --out "$OUT/pi05_${1}_partial_map_${a}${sfx}.csv" \
    --model pi0.5 --dataset "$1" --attack "$a" --attack-strength "$s" \
    --obs partial --obs-ratio "$3" --recovery map --null-count 32
}

LIB=/workspace/scratch/anon/libero10_wm_postprocess_full
RT=/workspace/scratch/anon/robotwin2/wm_eval
for tag in none clip_0.5 clip_1.0 clip_2.0 smooth_0.2 smooth_0.5 smooth_0.8 jitter_0.005 jitter_0.01 jitter_0.05 jitter_0.1; do
  run libero_10 "$LIB/$tag/$tag/task_rollout" 0.21875 "$tag"
  run robotwin10 "$RT/$tag" 0.4375 "$tag"
done

echo "[$(date +%H:%M:%S)] DONE openpi strength sweep"
ls "$OUT"/*.csv | wc -l
