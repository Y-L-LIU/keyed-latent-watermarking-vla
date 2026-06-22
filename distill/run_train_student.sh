#!/usr/bin/env bash
# Distill a student: norm-stats + LoRA fine-tune on the relabeled (teacher) corpus.
#   CFG=pi05_libero_goal_lora_distill_obstied  EXP=distill_obstied_k42  GPUS=0,1,2,3
set -uo pipefail
cd /workspace/vla/openpi
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache
export JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled
PY=/usr/bin/python3.11

CFG=${CFG:-pi05_libero_goal_lora_distill_obstied}
EXP=${EXP:-distill_obstied_k42}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=$GPUS   # scope norm-stats + train to this GPU set
LOGD=/workspace/vla/distill/logs; mkdir -p "$LOGD"
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

# Seed LIBERO config non-interactively if missing (needed for LiberoHdf5Dataset benchmark lookup).
if [ ! -f "$HOME/.libero/config.yaml" ]; then printf 'N\n' | $PY -c "import libero.libero" >/dev/null 2>&1 || true; fi

stamp "NORMSTATS $CFG start"
if $PY -m scripts.compute_norm_stats --config-name="$CFG" > "$LOGD/normstats_$CFG.log" 2>&1; then
  stamp "NORMSTATS $CFG OK"
else
  stamp "NORMSTATS $CFG FAIL -> $LOGD/normstats_$CFG.log"; tail -5 "$LOGD/normstats_$CFG.log"; exit 1
fi

stamp "TRAIN $CFG exp=$EXP gpus=$GPUS start"
$PY -m scripts.train "$CFG" --exp-name="$EXP" --overwrite \
  > "$LOGD/train_$CFG.log" 2>&1
rc=$?; stamp "TRAIN $CFG exit=$rc"
echo "STUDENT_TRAIN_COMPLETE cfg=$CFG exp=$EXP rc=$rc ckpt=/workspace/vla/openpi-checkpoints/$CFG/$EXP/2499"
exit $rc
