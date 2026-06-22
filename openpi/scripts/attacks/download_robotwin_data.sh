#!/bin/bash
set -e
cd /workspace/scratch/anon/robotwin2/hf_downloads

TASKS=(stack_bowls_two place_a2b_right open_laptop press_stapler place_shoe handover_block click_bell place_phone_stand stack_blocks_two)

download_one() {
    local t=$1
    local out="${t}_clean_50.zip"
    if [ -f "$out" ] && [ $(stat -c '%s' "$out") -gt 100000000 ]; then
        echo "skip $out (already $(stat -c '%s' "$out") bytes)"
        return
    fi
    local url="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset/${t}/aloha-agilex_clean_50.zip"
    echo "[$(date)] downloading $t"
    echo "[$(date)] done $t ($(stat -c '%s' "$out") bytes)"
}

# 3 parallel
PIDS=()
for t in "${TASKS[@]:0:3}"; do download_one "$t" & PIDS+=($!); done
wait "${PIDS[@]}"
PIDS=()
for t in "${TASKS[@]:3:3}"; do download_one "$t" & PIDS+=($!); done
wait "${PIDS[@]}"
PIDS=()
for t in "${TASKS[@]:6:3}"; do download_one "$t" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "all done"
