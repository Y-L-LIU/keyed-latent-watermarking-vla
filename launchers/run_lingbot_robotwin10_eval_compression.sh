#!/usr/bin/env bash
# CORRECTED lingbot-VA RoboTwin compression-robustness eval, matching the MAIN
# experiment recipe (attack_c_data/campaign/scripts/sweep_lingbot_robotwin.sh):
#   - INLINE-MAP2 worktree (main-branch post-hoc MAP is broken -> robotwin AUC~0.5)
#   - 10 robotwin tasks (NOT just beat_block_hammer)
#   - --chunk-period 2 (period 6 leaves ~22% of short episodes with 0 watermark windows)
#   - --map-num-starts 4 --map-optimizer adam (multi-start; single-start was wrong)
# Suspect = prune30 / int8-quant compressed descendant (single-server: suspect=detector).
# 4 GPUs (default 4-7): prune wm/plain on 4/5, quant wm/plain on 6/7; each loops 10 tasks.
set -uo pipefail
WT=/workspace/vla/lingbot-va
PY=/usr/bin/python3.11
export PYTHONPATH="$WT:/workspace/vla/openpi/third_party/libero:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd "$WT"
ROBOTWIN_ROOT=/workspace/vla/RoboTwin
OUTROOT=/workspace/vla/eval_out_compression
LOGD=/workspace/vla/eval_logs; mkdir -p "$LOGD"
TEST_NUM=${1:-10}   # episodes per task; 10 tasks x 10 = 100 per (attack,side)
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

TASKS=(adjust_bottle stack_bowls_two place_a2b_right open_laptop press_stapler \
       place_shoe handover_block click_bell place_phone_stand stack_blocks_two)
MAP_COMMON="--map-steps 10 --map-iters 30 --map-num-starts 4 --map-optimizer adam --skip-expert-check --chunk-period 2"

run_side(){   # $1=atk $2=side $3=gpu $4=beta $5=extra
  local atk=$1 side=$2 gpu=$3 beta=$4 extra=$5
  local cfg="robotwin_descendant_${atk}"
  local out="$OUTROOT/lingbot_robotwin10_${atk}_${side}"
  mkdir -p "$out"
  for t in "${TASKS[@]}"; do
    local port=$((29700 + gpu*11 + RANDOM%11))
    stamp "RT10/$atk/$side GPU$gpu task=$t beta=$beta test_num=$TEST_NUM"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      wan_va/wm/eval_robotwin_watermark.py \
      --config-name "$cfg" --robotwin-root "$ROBOTWIN_ROOT" --task-names "$t" \
      --test-num "$TEST_NUM" --beta "$beta" $MAP_COMMON $extra \
      --out-dir "$out" \
      >> "$LOGD/eval_lingbot_rt10_${atk}_${side}_g${gpu}.log" 2>&1
    stamp "RT10/$atk/$side task=$t done exit=$?"
  done
}

GPU_PRUNE_WM=${GPU_PRUNE_WM:-4}; GPU_PRUNE_PL=${GPU_PRUNE_PL:-5}
GPU_QUANT_WM=${GPU_QUANT_WM:-6}; GPU_QUANT_PL=${GPU_QUANT_PL:-7}

# SYMMETRY FIX: plain MUST also run MAP (4-start), else wm(MAP) vs plain(no-MAP) is
# apples-vs-oranges and the AUC is untrustworthy. Removed --skip-map from plain.
run_side prune wm    $GPU_PRUNE_WM 1.0 "" &
run_side prune plain $GPU_PRUNE_PL 0.0 "" &
run_side quant wm    $GPU_QUANT_WM 1.0 "" &
run_side quant plain $GPU_QUANT_PL 0.0 "" &
wait
stamp "ALL lingbot RT10 compression eval done"
echo "LINGBOT_RT_COMP_EVAL_COMPLETE"
