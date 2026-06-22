#!/bin/bash
# Lingbot robotwin full sweep (INLINE-FIXED worktree). Resumable: the eval skips
# episodes whose npz already exists, so re-running tops up to a higher TESTNUM and
# fills the new attack conditions. 4 MAP starts / 30 iters / 10 decode.
# Dispatches one (condition,task) job per free GPU.
set -uo pipefail
WT=/workspace/vla/lingbot-va
PY=/workspace/vla/.venv/bin/torchrun
ROBOTWIN_ROOT=/workspace/vla/RoboTwin
OUT=/workspace/vla/attack_c_data/rollouts/lingbot_robotwin_p2
LOGDIR=/root/campaign_logs/lingbot_robotwin_p2
mkdir -p "$LOGDIR"
cd "$WT"; export PYTHONPATH="$WT"; export PYTHONDONTWRITEBYTECODE=1

TESTNUM=${TESTNUM:-30}
GPUS=(${GPUS:-0 1 2 3})
TASKS=(adjust_bottle stack_bowls_two place_a2b_right open_laptop press_stapler \
       place_shoe handover_block click_bell place_phone_stand stack_blocks_two)
MAP_COMMON="--map-steps 10 --map-iters 30 --map-num-starts 4 --map-optimizer adam --skip-expert-check --chunk-period 2"
BASE=wan_va/wm/eval_robotwin_watermark.py
ROB=wan_va/wm/eval_robotwin_watermark_robustness.py

# condition spec: "name|eval|out_subdir|extra_args"
CONDS=(
  "normal|$BASE|$OUT/normal|--beta 1.0"
  "plain|$BASE|$OUT/plain|--beta 0.0 --skip-map"
  "clip_0.5|$ROB|$OUT/robust|--controller-postprocess clip --controller-clip-limit 0.5"
  "clip_1.0|$ROB|$OUT/robust|--controller-postprocess clip --controller-clip-limit 1.0"
  "clip_2.0|$ROB|$OUT/robust|--controller-postprocess clip --controller-clip-limit 2.0"
  "smooth_0.3|$ROB|$OUT/robust|--controller-postprocess smooth --controller-smooth-alpha 0.3"
  "smooth_0.5|$ROB|$OUT/robust|--controller-postprocess smooth --controller-smooth-alpha 0.5"
  "smooth_0.7|$ROB|$OUT/robust|--controller-postprocess smooth --controller-smooth-alpha 0.7"
  "jitter_0.005|$ROB|$OUT/robust|--controller-postprocess jitter --controller-jitter-std 0.005"
  "jitter_0.01|$ROB|$OUT/robust|--controller-postprocess jitter --controller-jitter-std 0.01"
  "jitter_0.02|$ROB|$OUT/robust|--controller-postprocess jitter --controller-jitter-std 0.02"
  "delay_1|$ROB|$OUT/robust|--controller-postprocess delay --controller-delay-steps 1"
  "delay_2|$ROB|$OUT/robust|--controller-postprocess delay --controller-delay-steps 2"
  "delay_3|$ROB|$OUT/robust|--controller-postprocess delay --controller-delay-steps 3"
)

# Build flat job list
JOBS=()
for c in "${CONDS[@]}"; do
  for t in "${TASKS[@]}"; do JOBS+=("$c|$t"); done
done
echo "[$(date +%T)] lingbot robotwin sweep: ${#JOBS[@]} jobs, TESTNUM=$TESTNUM, GPUs=${GPUS[*]}"

declare -A PIDS
launch() {  # $1=gpu $2=jobspec
  local gpu=$1 spec=$2
  IFS='|' read -r name eval outdir extra task <<< "$spec"
  local port=$((29600 + gpu*7 + RANDOM%7))
  local log="$LOGDIR/${name}_${task}.log"
  CUDA_VISIBLE_DEVICES=$gpu $PY --nproc_per_node=1 --master_port=$port \
    "$eval" --robotwin-root "$ROBOTWIN_ROOT" --task-names "$task" \
    --test-num "$TESTNUM" $MAP_COMMON $extra --out-dir "$outdir" \
    > "$log" 2>&1 &
  PIDS[$gpu]=$!
}

idx=0
# prime
for g in "${GPUS[@]}"; do
  [ $idx -lt ${#JOBS[@]} ] && { launch "$g" "${JOBS[$idx]}"; echo "[$(date +%T)] GPU$g <- ${JOBS[$idx]%%|*}/$(echo ${JOBS[$idx]}|awk -F'|' '{print $5}')"; idx=$((idx+1)); }
done
# dispatch loop
while [ $idx -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    pid=${PIDS[$g]:-}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      unset PIDS[$g]
      if [ $idx -lt ${#JOBS[@]} ]; then
        launch "$g" "${JOBS[$idx]}"; echo "[$(date +%T)] GPU$g <- ${JOBS[$idx]%%|*}/$(echo ${JOBS[$idx]}|awk -F'|' '{print $5}')"; idx=$((idx+1))
      fi
    fi
  done
  sleep 15
done
echo "[$(date +%T)] lingbot robotwin sweep DONE"
