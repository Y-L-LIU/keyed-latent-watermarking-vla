#!/usr/bin/env bash
# Faithful latent-DC: wait for relabel -> train student (params-only) -> roll out plain
# via the eval (MAP saves recovered seed) -> latent-DC detector vs clean-control student.
set -uo pipefail
DISTILL=/workspace/vla/distill; LOGD=$DISTILL/logs
CKPT=/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_latentdc/distill_latentdc_k42/1499
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "ORCH-LDC: waiting for latent-DC relabel (4 shards)..."
until [ "$(grep -l DONE $LOGD/relabel_latentdc_shard*.log 2>/dev/null | wc -l)" -eq 4 ]; do sleep 15; done
stamp "ORCH-LDC: relabel done; training latent-DC student (params-only, GPUs 0-3)"
OPENPI_PARAMS_ONLY=1 CFG=pi05_libero_goal_lora_distill_latentdc EXP=distill_latentdc_k42 GPUS=0,1,2,3 \
  bash $DISTILL/run_train_student.sh > $LOGD/launch_latentdc.log 2>&1
until [ -d "$CKPT/params" ]; do sleep 15; done
sleep 10
stamp "ORCH-LDC: rolling out student plain (MAP) sharded GPUs 0-3"
OFFS=(0 3 6 8); CNTS=(3 3 2 2)
for s in 0 1 2 3; do
  OUT=$DISTILL/eval GPU=$s TAG=latentdc_student TASKS=${CNTS[$s]} OFFSET=${OFFS[$s]} TRIALS=5 \
    POLICY_CFG=pi05_libero_goal_lora_distill_latentdc POLICY_CKPT=$CKPT \
    DET_CFG=pi05_libero DET_CKPT=/workspace/vla/models/pi05_libero KEYING=observation \
    nohup bash $DISTILL/run_eval_obstied.sh > "$LOGD/eval_latentdc_shard${s}.log" 2>&1 &
done
wait
stamp "ORCH-LDC: scoring latent-DC detector"
export PYTHONPATH=/workspace/vla/distill:/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src
/usr/bin/python3.11 $DISTILL/score_latentdc.py \
  --latentdc-rollouts $DISTILL/eval/libero_goal_latentdc_student/rollouts/task_rollout \
  --clean-rollouts   $DISTILL/eval/libero_goal_clean_student/rollouts/task_rollout \
  --secret-key 42 2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprec|flax" | tee $DISTILL/VERDICT_latentdc.txt
# free the checkpoint promptly (disk)
rm -rf /workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_latentdc
stamp "ORCH-LDC: DONE (ckpt deleted) -> $DISTILL/VERDICT_latentdc.txt"
echo "ORCH_LATENTDC_COMPLETE"
