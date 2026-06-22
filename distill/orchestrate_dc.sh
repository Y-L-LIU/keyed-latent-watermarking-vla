#!/usr/bin/env bash
# Positive control: wait for DC relabel -> train DC student (params-only) -> retention test.
# A learnable + temporally-persistent (DC per-task) offset SHOULD survive distillation.
set -uo pipefail
DISTILL=/workspace/vla/distill; LOGD=$DISTILL/logs
CKPT=/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_outputtaskdc/distill_outputtaskdc_k42/1499
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "ORCH-DC: waiting for DC relabel (4 shards)..."
until [ "$(grep -l DONE $LOGD/relabel_dc_shard*.log 2>/dev/null | wc -l)" -eq 4 ]; do sleep 15; done
stamp "ORCH-DC: relabel done; training DC student (params-only, GPUs 0-3)"
OPENPI_PARAMS_ONLY=1 CFG=pi05_libero_goal_lora_distill_outputtaskdc EXP=distill_outputtaskdc_k42 GPUS=0,1,2,3 \
  bash $DISTILL/run_train_student.sh > $LOGD/launch_dc.log 2>&1
until [ -d "$CKPT/params" ]; do sleep 15; done
sleep 10
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi/third_party/libero
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=disabled
cd /workspace/vla/openpi
stamp "ORCH-DC: bias-retention test (DC per-task, learnable+persistent -> positive control)"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.11 $DISTILL/bias_retention_test.py \
  --student-config pi05_libero_goal_lora_distill_outputtaskdc --student-ckpt "$CKPT" \
  --keying output_task_dc --secret-key 42 --beta-out 0.1 \
  2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprecation|flax|swigvarlink|using task" | tee $DISTILL/VERDICT_dc.txt
stamp "ORCH-DC: DONE -> $DISTILL/VERDICT_dc.txt"
echo "ORCH_DC_COMPLETE"
