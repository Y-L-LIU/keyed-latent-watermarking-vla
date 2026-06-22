#!/usr/bin/env bash
# Assemble a server-loadable ckpt dir (base vae/text_encoder/tokenizer/assets + student
# transformer/) via symlinks, then run the retention test for one arm.
# Env: ARM=dc|hash  [N_EPS=40]
set -uo pipefail
cd /workspace/vla/lingbot-va
export PYTHONPATH=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/vla/lingbot_latents
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM=${ARM:?set ARM=dc|hash}
BASE=/workspace/vla/models/lingbot-va-posttrain-libero-long
SAVE_ROOT=${SAVE_ROOT:-/workspace/vla/lingbot_out/distill_ckpt/student_${ARM}}
# pick the latest checkpoint_step_* dir
CKPT_STEP=$(ls -d ${SAVE_ROOT}/checkpoints/checkpoint_step_* 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_STEP" ]; then echo "NO CKPT in ${SAVE_ROOT}/checkpoints"; exit 2; fi
echo "[assemble] using ${CKPT_STEP}"

ASM=${ASM:-${SAVE_ROOT%/*}/assembled_${ARM}}
rm -rf "$ASM"; mkdir -p "$ASM"
for d in vae text_encoder tokenizer assets README.md; do
  ln -s "$BASE/$d" "$ASM/$d"
done
ln -s "$CKPT_STEP/transformer" "$ASM/transformer"
echo "[assemble] assembled -> $ASM"
ls -la "$ASM"

PY=/usr/bin/python3.11
GPU=${GPU:-2}
N_EPS=${N_EPS:-40}
LOG=/workspace/vla/ft_logs/retn_${ARM}.log
echo "[$(date +%H:%M:%S)] retention ${ARM} start (gpu${GPU}) n_eps=${N_EPS}" | tee $LOG
CUDA_VISIBLE_DEVICES=$GPU STUDENT_CKPT="$ASM" ARM=$ARM N_EPS=$N_EPS N_KEYS=${N_KEYS:-0} PERSTEP=${PERSTEP:-0} \
  OUT_JSON=${OUT_JSON:-/workspace/vla/distill/lingbot/retention_${ARM}.json} \
  $PY -m torch.distributed.run --nproc_per_node=1 --master_port 29532 \
  /workspace/vla/distill/lingbot/retention_test.py >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] retention ${ARM} exit=$?" | tee -a $LOG
echo "RETN_${ARM}_DONE" | tee -a $LOG
