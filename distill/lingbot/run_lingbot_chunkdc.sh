#!/usr/bin/env bash
# CORRECTED LingBot entropy sweep (faithful pi0.5 analog). chunkdc = per H-aligned chunk a
# CONSTANT beta-scaled dc_offset keyed on the chunk-START observation bucket mod N_KEYS:
# a non-zero-mean function of the conditioning the policy sees -> learnable by BC (the
# per-timestep hashmod keyed on unobserved within-chunk states was averaged to 0). Detect
# with the per-chunk (first-frame, PERSTEP=0) reference -> ceiling 1.0. Reuses student_dc
# (per-task ~10, retention 0.478) as the low-entropy anchor. KEEPS ckpts under /workspace/vla_out.
set -uo pipefail
LB=/workspace/vla/distill/lingbot
CKROOT=/workspace/vla_out
stamp(){ echo "[$(date +%H:%M:%S)] LB-CHUNK $*"; }
export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/distill:${PYTHONPATH:-}
declare -A NK=( [chunkdc40]=40 [chunkdc160]=160 )

for ARM in chunkdc40 chunkdc160; do
  N=${NK[$ARM]}
  DS=/workspace/vla/lingbot_latents/relabel_${ARM}
  SAVE=$CKROOT/student_${ARM}
  stamp "=== $ARM start (N_KEYS=$N) ==="
  if [ "$(ls $DS/data/chunk-000/*.parquet 2>/dev/null | wc -l)" -lt 500 ]; then stamp "$ARM: relabel missing"; exit 1; fi
  if [ -d "$SAVE/checkpoints/checkpoint_step_1500/transformer" ]; then
    stamp "$ARM: ckpt exists, skip train"
  else
    stamp "$ARM: train -> $SAVE"
    ARM=$ARM DATASET_PATH=$DS SAVE_ROOT=$SAVE NUM_STEPS=1500 \
      bash $LB/launch_train.sh > /workspace/vla/ft_logs/launch_${ARM}.log 2>&1
    if ! grep -q "STUDENT_${ARM}_DONE" /workspace/vla/ft_logs/train_student_${ARM}.log 2>/dev/null; then
      stamp "$ARM: train FAILED"; tail -6 /workspace/vla/ft_logs/train_student_${ARM}.log; exit 1
    fi
  fi
  stamp "$ARM: per-chunk retention (PERSTEP=0)"
  ARM=$ARM SAVE_ROOT=$SAVE N_KEYS=$N PERSTEP=0 GPU=4 N_EPS=40 \
    OUT_JSON=$LB/retention_${ARM}.json \
    bash $LB/assemble_and_retn.sh
  [ -f "$LB/retention_${ARM}.json" ] || { stamp "$ARM: no json"; exit 1; }
  stamp "=== $ARM DONE (ckpt KEPT) ==="
done

stamp "ALL DONE"
echo "=== CORRECTED LingBot entropy curve (chunk-conditioning keyed) ==="
echo "  dc (per-task ~10): $(/usr/bin/python3.11 -c "import json;print(round(json.load(open('$LB/retention_dc.json'))['retention'],3))")"
for ARM in chunkdc40 chunkdc160; do
  echo "  $ARM: $(/usr/bin/python3.11 -c "import json;d=json.load(open('$LB/retention_${ARM}.json'));print('retn',round(d['retention'],3),'survives',d['survives'],'Z',round(d['z'],2))")"
done
echo "LINGBOT_CHUNKDC_COMPLETE"
