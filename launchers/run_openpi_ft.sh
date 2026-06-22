#!/usr/bin/env bash
# openpi LoRA fine-tune: two parallel 4-GPU jobs (LIBERO goal on 0-3, spatial on 4-7).
# Produces "fine-tuned descendant" checkpoints for the watermark robustness scenario (§12.5).
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # GPU0 has ~9GB leaked from a dead process; 0.85 stays under free
export WANDB_MODE=disabled
PY=/usr/bin/python3.11

# Seed LIBERO config non-interactively if missing (default datasets path = our submodule dir).
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  printf 'N\n' | $PY -c "import libero.libero" >/dev/null 2>&1 || true
fi
LOGD=/workspace/vla/ft_logs; mkdir -p "$LOGD"
EXP=descendant_lora
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

GOAL=pi05_libero_goal_lora_from_libero
SPAT=pi05_libero_spatial_lora_from_libero

# 1) norm stats (sequential, cheap; required before training)
for cfg in "$GOAL" "$SPAT"; do
  stamp "NORMSTATS $cfg start"
  if $PY -m scripts.compute_norm_stats --config-name="$cfg" > "$LOGD/normstats_$cfg.log" 2>&1; then
    stamp "NORMSTATS $cfg OK"
  else
    stamp "NORMSTATS $cfg FAIL (see $LOGD/normstats_$cfg.log)"
  fi
done

# 2) train both in parallel on disjoint GPU sets
stamp "TRAIN start: goal(gpu0-3) + spatial(gpu4-7)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m scripts.train "$GOAL" --exp-name="$EXP" --overwrite > "$LOGD/train_goal.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY -m scripts.train "$SPAT" --exp-name="$EXP" --overwrite > "$LOGD/train_spatial.log" 2>&1 &
P2=$!
wait $P1; R1=$?; stamp "TRAIN goal exit=$R1"
wait $P2; R2=$?; stamp "TRAIN spatial exit=$R2"

stamp "ALL TRAINING DONE goal_exit=$R1 spatial_exit=$R2"
echo "OPENPI_FT_COMPLETE goal=$R1 spatial=$R2"
