#!/usr/bin/env bash
cd /workspace/vla/lingbot-va
export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=osmesa WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=/usr/bin/python3.11
launch(){ local kind=$1 g=$2 t0=$3 t1=$4 port=$5 beta=$6 outdir=$7
  echo "[$(date +%H:%M:%S)] launch $kind GPU$g tasks $t0-$t1 port $port"
  CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
    -m wan_va.wm.eval_libero_watermark \
    --config-name libero_descendant --suite libero_10 \
    --task-range $t0 $t1 --test-num 5 \
    --beta $beta --chunk-period 6 --chunk-start-min 2 \
    --map-iters 30 --map-steps 10 --map-lr 0.08 \
    --out-dir $outdir \
    > /workspace/vla/eval_logs/eval_lingbot_libero10_${kind}_g${g}.log 2>&1
  echo "[$(date +%H:%M:%S)] $kind GPU$g done exit=$?"
}
launch wm    4 0  5  29557 1.0 /workspace/vla/eval_out/lingbot_libero10_descendant       &
launch wm    5 5 10 29558 1.0 /workspace/vla/eval_out/lingbot_libero10_descendant       &
launch plain 6 0  5 29559 0.0 /workspace/vla/eval_out/lingbot_libero10_descendant_plain &
launch plain 7 5 10 29560 0.0 /workspace/vla/eval_out/lingbot_libero10_descendant_plain &
wait
echo "LINGBOT_FULL_EVAL_COMPLETE at $(date +%H:%M:%S)"
