#!/usr/bin/env bash
# Autonomous: wait for both student trainings, eval each student (plain+watermarked)
# sharded across 4 GPUs, then run the distillation-survival analyzer -> VERDICT.
set -uo pipefail
DISTILL=/workspace/vla/distill
LOGD=$DISTILL/logs
CKPT_ROOT=/workspace/vla/openpi-checkpoints
OBST_CKPT=$CKPT_ROOT/pi05_libero_goal_lora_distill_obstied/distill_obstied_k42/2499
CLEAN_CKPT=$CKPT_ROOT/pi05_libero_goal_lora_distill_clean/distill_clean_k42/2499
TRIALS=${TRIALS:-5}
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "ORCH: waiting for both student checkpoints..."
while true; do
  if [ -d "$OBST_CKPT/params" ] && [ -d "$CLEAN_CKPT/params" ]; then
    stamp "ORCH: both checkpoints present"; break
  fi
  sleep 30
done
sleep 10  # let checkpoint writer finish

# 10 libero_goal tasks -> 4 shards: offsets/counts
OFFS=(0 3 6 8); CNTS=(3 3 2 2)

run_student(){
  local cfg=$1 ckpt=$2 tag=$3 gpubase=$4
  for s in 0 1 2 3; do
    local gpu=$((gpubase + s))
    OUT=$DISTILL/eval GPU=$gpu TAG=$tag TASKS=${CNTS[$s]} OFFSET=${OFFS[$s]} TRIALS=$TRIALS \
      POLICY_CFG=$cfg POLICY_CKPT=$ckpt DET_CFG=pi05_libero DET_CKPT=/workspace/vla/models/pi05_libero \
      SECRET_KEY=42 Q=0.08 PROJ=0,1,2 KEYING=observation \
      nohup bash $DISTILL/run_eval_obstied.sh > "$LOGD/eval_${tag}_shard${s}.log" 2>&1 &
  done
}

stamp "ORCH: launching obstied-student eval (GPUs 0-3) + clean-student eval (GPUs 4-7)"
run_student pi05_libero_goal_lora_distill_obstied "$OBST_CKPT" obstied_student 0
run_student pi05_libero_goal_lora_distill_clean  "$CLEAN_CKPT" clean_student   4
wait
stamp "ORCH: all eval shards done"

stamp "ORCH: running analyzer"
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src
/usr/bin/python3.11 $DISTILL/analyze_distill.py \
  --obstied-rollouts $DISTILL/eval/libero_goal_obstied_student/rollouts/task_rollout \
  --clean-rollouts   $DISTILL/eval/libero_goal_clean_student/rollouts/task_rollout \
  --teacher-rollouts $DISTILL/phase1_smoke/libero_goal_teacher_smoke/rollouts/task_rollout \
  --secret-key 42 --q 0.08 --proj-dims 0,1,2 2>&1 | grep -vE "WARNING|warn|tcmalloc" \
  | tee $DISTILL/VERDICT.txt
stamp "ORCH: DONE -> $DISTILL/VERDICT.txt"
echo "ORCH_COMPLETE"
