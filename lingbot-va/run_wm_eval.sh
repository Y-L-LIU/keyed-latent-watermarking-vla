#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myuv
cd ~/projects/vla/lingbot-va

export CUDA_VISIBLE_DEVICES=1,2
export MUJOCO_GL=osmesa

echo "[$(date)] Starting libero_goal..."
python -m wan_va.wm.eval_libero_watermark --config-name libero --suite libero_goal --num-gpus 2 --test-num 10 --out-dir outputs/wm_libero 2>&1
echo "[$(date)] Finished libero_goal."

echo "[$(date)] Starting libero_spatial..."
python -m wan_va.wm.eval_libero_watermark --config-name libero --suite libero_spatial --num-gpus 2 --test-num 10 --out-dir outputs/wm_libero 2>&1
echo "[$(date)] Finished libero_spatial."

echo "[$(date)] Starting libero_object..."
python -m wan_va.wm.eval_libero_watermark --config-name libero --suite libero_object --num-gpus 2 --test-num 10 --out-dir outputs/wm_libero 2>&1
echo "[$(date)] Finished libero_object."

echo "[$(date)] ALL DONE."
