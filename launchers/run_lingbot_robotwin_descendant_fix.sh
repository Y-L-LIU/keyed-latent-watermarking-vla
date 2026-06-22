#!/usr/bin/env bash
# Fix lingbot/RoboTwin DESCENDANT cell: re-run on the correct robotwin10 task set
# (NOT beat_block_hammer) with BOTH variants going through MAP (no --skip-map), so the
# verification/identification comparison is symmetric. Matches the wm MAP config used by
# the original n100 (adam, iters30, steps10, lr0.08, num_starts4). Pull queue over GPUs 2-7.
set -uo pipefail
cd /workspace/vla/lingbot-va
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/vla/lingbot-va:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
PY=/usr/bin/python3.11
OUTROOT=/workspace/vla/eval_out/lingbot_rt_descendant_fix10
LOGD=/workspace/vla/eval_logs/lingbot_fix
QDIR="$LOGD/queue"
mkdir -p "$OUTROOT" "$LOGD"

TEST_NUM=${TEST_NUM:-13}
GPUS=(${GPUS:-2 3 4 5 6 7})
# lingbot robotwin10 tasks (the set used by the clean + attack cells)
TASKS=(adjust_bottle click_bell handover_block open_laptop place_a2b_right \
       place_phone_stand place_shoe press_stapler stack_blocks_two stack_bowls_two)
# variant -> beta
declare -A BETA=( [wm]=1.0 [plain]=0.0 )

# jobs = (task,variant); BOTH variants get MAP (no --skip-map)
JOBS=()
for v in wm plain; do for t in "${TASKS[@]}"; do JOBS+=("$t|$v"); done; done
TOTAL=${#JOBS[@]}

rm -rf "$QDIR"; mkdir -p "$QDIR"; echo 0 > "$QDIR/counter"
claim(){ exec 9>"$QDIR/lock"; flock 9; local i; i=$(cat "$QDIR/counter"); echo $((i+1))>"$QDIR/counter"; flock -u 9; echo "$i"; }
stamp(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/dispatch_fix10.log"; }

worker(){
  local gpu=$1
  while true; do
    local idx; idx=$(claim)
    if [ "$idx" -ge "$TOTAL" ]; then break; fi
    local job="${JOBS[$idx]}" task variant
    task="${job%|*}"; variant="${job#*|}"
    local beta="${BETA[$variant]}"
    local port=$((29600 + gpu*13 + idx%13))
    local out="$OUTROOT/$task/$variant"
    local log="$LOGD/gpu${gpu}__${task}__${variant}.log"
    mkdir -p "$out"
    stamp "gpu=$gpu idx=$idx START task=$task variant=$variant beta=$beta -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run \
      --nproc_per_node=1 --master_port=$port \
      wan_va/wm/eval_robotwin_watermark.py \
      --config-name robotwin_descendant --robotwin-root /workspace/vla/RoboTwin \
      --task-names "$task" --test-num "$TEST_NUM" \
      --secret-key 17 --beta "$beta" --chunk-period 2 --chunk-start-min 2 \
      --map-iters 30 --map-steps 10 --map-lr 0.08 --map-prior-weight 1.0 \
      --map-optimizer adam --map-num-starts 4 --skip-expert-check \
      --out-dir "$out" \
      > "$log" 2>&1
    stamp "gpu=$gpu idx=$idx DONE  task=$task variant=$variant exit=$?"
  done
  stamp "gpu=$gpu worker drained"
}

stamp "==== LINGBOT RT DESCENDANT FIX START: $TOTAL jobs (${#TASKS[@]} tasks x {wm,plain}), TEST_NUM=$TEST_NUM, GPUs=${GPUS[*]} ===="
for g in "${GPUS[@]}"; do worker "$g" & done
wait
stamp "==== ALL LINGBOT FIX JOBS DONE ===="
echo "LINGBOT_RT_DESCENDANT_FIX_COMPLETE"
