#!/usr/bin/env bash
# Autonomous: wait for task-tied relabel -> train task student (params-only, GPUs 4-7)
# -> bias-retention test on BOTH the task (learnable) and hash (unlearnable) students.
set -uo pipefail
DISTILL=/workspace/vla/distill; LOGD=$DISTILL/logs
TASK_CKPT=/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_outputtask/distill_outputtask_k42/1499
HASH_CKPT=/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_output/distill_output_k42/1499
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "ORCH-TASK: waiting for task-tied relabel (4 shards)..."
until [ "$(grep -l DONE $LOGD/relabel_outputtask_shard*.log 2>/dev/null | wc -l)" -eq 4 ]; do sleep 15; done
stamp "ORCH-TASK: relabel done; training task student (params-only, GPUs 4-7)"

OPENPI_PARAMS_ONLY=1 CFG=pi05_libero_goal_lora_distill_outputtask EXP=distill_outputtask_k42 GPUS=4,5,6,7 \
  bash $DISTILL/run_train_student.sh > $LOGD/launch_outputtask.log 2>&1
stamp "ORCH-TASK: training done; waiting for checkpoint"
until [ -d "$TASK_CKPT/params" ]; do sleep 15; done
sleep 10

export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi/third_party/libero
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=disabled
cd /workspace/vla/openpi
stamp "ORCH-TASK: bias-retention test (task student, learnable key)"
CUDA_VISIBLE_DEVICES=4 /usr/bin/python3.11 $DISTILL/bias_retention_test.py \
  --student-config pi05_libero_goal_lora_distill_outputtask --student-ckpt "$TASK_CKPT" \
  --keying output_task --secret-key 42 --beta-out 0.1 \
  2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprecation|flax|swigvarlink|using task" | tee $DISTILL/VERDICT_outputtask.txt
echo "" | tee -a $DISTILL/VERDICT_outputtask.txt
stamp "ORCH-TASK: bias-retention test (hash student, unlearnable key) for contrast"
CUDA_VISIBLE_DEVICES=4 /usr/bin/python3.11 $DISTILL/bias_retention_test.py \
  --student-config pi05_libero_goal_lora_distill_output --student-ckpt "$HASH_CKPT" \
  --keying output --secret-key 42 --beta-out 0.1 \
  2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprecation|flax|swigvarlink|using task" | tee -a $DISTILL/VERDICT_outputtask.txt
stamp "ORCH-TASK: DONE -> $DISTILL/VERDICT_outputtask.txt"
echo "ORCH_TASK_COMPLETE"
