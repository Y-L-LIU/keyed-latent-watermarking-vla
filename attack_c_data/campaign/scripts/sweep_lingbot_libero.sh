#!/bin/bash
# Lingbot libero_10 batch (inline-fixed worktree, torch.load patched). Clean + plain
# baseline + 4 attacks x1 canonical strength. Resumable (eval skips existing npz).
# One condition = one job (all 10 tasks internally). Dispatches across GPU pool;
# safe to re-run with a bigger pool later (skip-exists fills the rest).
set -uo pipefail
WT=/workspace/vla/lingbot-va
LIBERO=/workspace/vla/openpi/third_party/libero
PATCH=/workspace/vla/openpi/scripts/attacks/run_eval_with_libero_patch.py
PY=/workspace/vla/.venv/bin/torchrun
OUT=/workspace/vla/attack_c_data/rollouts/lingbot_libero
LOGDIR=/root/campaign_logs/lingbot_libero
mkdir -p "$LOGDIR" "$OUT"
cd "$WT"; export PYTHONPATH="$WT:$LIBERO"; export PYTHONDONTWRITEBYTECODE=1; export MUJOCO_GL=egl

TESTNUM=${TESTNUM:-12}
GPUS=(${GPUS:-7})
BASE=wan_va/wm/eval_libero_watermark.py
ROB=wan_va/wm/eval_libero_watermark_robustness.py
MAP_COMMON="--map-steps 10 --map-iters 30"

# "name|eval|out_subdir|extra_args"
CONDS=(
  "normal|$BASE|$OUT/normal|--beta 1.0"
  "plain|$BASE|$OUT/plain|--beta 0.0 --skip-map"
  "clip_1.0|$ROB|$OUT/robust|--controller-postprocess clip --controller-clip-limit 1.0"
  "smooth_0.5|$ROB|$OUT/robust|--controller-postprocess smooth --controller-smooth-alpha 0.5"
  "jitter_0.01|$ROB|$OUT/robust|--controller-postprocess jitter --controller-jitter-std 0.01"
  "delay_1|$ROB|$OUT/robust|--controller-postprocess delay --controller-delay-steps 1"
)
echo "[$(date +%T)] lingbot libero batch: ${#CONDS[@]} conditions, TESTNUM=$TESTNUM, GPUs=${GPUS[*]}"

declare -A PIDS
launch() {  # $1=gpu $2=cond
  local gpu=$1; IFS='|' read -r name eval outdir extra <<< "$2"
  local port=$((29680 + gpu*5 + RANDOM%5))
  MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu $PY --nproc_per_node=1 --master_port=$port \
    "$PATCH" "$eval" -- --suite libero_10 --test-num "$TESTNUM" $MAP_COMMON $extra --out-dir "$outdir" \
    > "$LOGDIR/${name}.log" 2>&1 &
  PIDS[$gpu]=$!
}

idx=0
for g in "${GPUS[@]}"; do
  [ $idx -lt ${#CONDS[@]} ] && { launch "$g" "${CONDS[$idx]}"; echo "[$(date +%T)] GPU$g <- ${CONDS[$idx]%%|*}"; idx=$((idx+1)); }
done
while [ $idx -lt ${#CONDS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    pid=${PIDS[$g]:-}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      unset PIDS[$g]
      if [ $idx -lt ${#CONDS[@]} ]; then launch "$g" "${CONDS[$idx]}"; echo "[$(date +%T)] GPU$g <- ${CONDS[$idx]%%|*}"; idx=$((idx+1)); fi
    fi
  done
  sleep 20
done
echo "[$(date +%T)] lingbot libero batch DONE"
