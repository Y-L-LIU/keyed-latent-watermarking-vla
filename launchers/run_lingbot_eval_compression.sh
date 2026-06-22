#!/usr/bin/env bash
# Lingbot-VA LIBERO compression-robustness eval (threat-model §12.5):
#   suspect = lingbot LIBERO LoRA descendant with prune30 OR int8 weight quant
#   detector = unmodified base detector (mainline scoring inside the eval)
# Splits tasks 0-9 across 2 GPUs per side (wm / plain) and runs both attacks
# in parallel = 8 GPUs total. test_num=10 -> 100 plain + 100 wm per attack.
set -uo pipefail
cd /workspace/vla/lingbot-va
export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=/usr/bin/python3.11
TEST_NUM=${1:-10}
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
OUTROOT=/workspace/vla/eval_out_compression
mkdir -p "$OUTROOT"
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

launch(){
  local atk=$1 side=$2 gpu=$3 t0=$4 t1=$5 port=$6 beta=$7
  local outdir="$OUTROOT/lingbot_libero10_${atk}_${side}"
  mkdir -p "$outdir"
  stamp "launch ${atk}/${side} GPU${gpu} tasks ${t0}-${t1} port ${port} beta=${beta}"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
    -m wan_va.wm.eval_libero_watermark \
    --config-name "libero_descendant_${atk}" --suite libero_10 \
    --task-range $t0 $t1 --test-num $TEST_NUM \
    --beta $beta --chunk-period 6 --chunk-start-min 2 \
    --map-iters 30 --map-steps 10 --map-lr 0.08 \
    --out-dir "$outdir" \
    > "$LOGD/eval_lingbot_${atk}_${side}_g${gpu}.log" 2>&1
  stamp "${atk}/${side} GPU${gpu} done exit=$?"
}

# prune30: GPUs 0(wm 0-5), 1(wm 5-10), 2(plain 0-5), 3(plain 5-10)
launch prune wm    0 0 5  29571 1.0 &
launch prune wm    1 5 10 29572 1.0 &
launch prune plain 2 0 5  29573 0.0 &
launch prune plain 3 5 10 29574 0.0 &
# quant:   GPUs 4(wm 0-5), 5(wm 5-10), 6(plain 0-5), 7(plain 5-10)
launch quant wm    4 0 5  29575 1.0 &
launch quant wm    5 5 10 29576 1.0 &
launch quant plain 6 0 5  29577 0.0 &
launch quant plain 7 5 10 29578 0.0 &
wait
stamp "ALL LINGBOT LIBERO COMPRESSION DONE"
echo "LINGBOT_COMP_LIBERO_COMPLETE"
