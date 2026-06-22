#!/usr/bin/env bash
# LingBot entropy curve with the CORRECT detector (per-step keying, ceiling=1.0).
# Retrains 4 students (dc / hashmod40 / hashmod160 / hash) from the EXISTING relabel
# datasets, KEEPS the checkpoints under /workspace/vla_out (persistent, NOT deleted), and measures
# retention with PERSTEP=1 so the obs-tied entropy gradient is resolvable (the first-frame
# metric capped the obs-tied ceiling at ~0.62 and was blind to cardinality).
# Sequential (~45min/arm on gpu4-7). Results -> retention_perstep_<arm>.json
set -uo pipefail
LB=/workspace/vla/distill/lingbot
CKROOT=/workspace/vla_out
mkdir -p "$CKROOT"
stamp(){ echo "[$(date +%H:%M:%S)] LB-PS $*"; }
export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/distill:${PYTHONPATH:-}

# arm -> N_KEYS (0 = dc / full-hash, no mod)
declare -A NK=( [dc]=0 [hashmod40]=40 [hashmod160]=160 [hash]=0 )

for ARM in dc hashmod40 hashmod160 hash; do
  DS=/workspace/vla/lingbot_latents/relabel_${ARM}
  SAVE=$CKROOT/student_${ARM}
  stamp "=== $ARM start (N_KEYS=${NK[$ARM]}) ==="
  if [ "$(ls $DS/data/chunk-000/*.parquet 2>/dev/null | wc -l)" -lt 500 ]; then
    stamp "$ARM: relabel dataset missing at $DS"; exit 1
  fi
  # 1. train (save to /workspace/vla_out, persistent)
  if [ -d "$SAVE/checkpoints/checkpoint_step_1500/transformer" ]; then
    stamp "$ARM: ckpt already exists, skip train"
  else
    stamp "$ARM: train -> $SAVE"
    ARM=$ARM DATASET_PATH=$DS SAVE_ROOT=$SAVE NUM_STEPS=1500 \
      bash $LB/launch_train.sh > /workspace/vla/ft_logs/launch_ps_${ARM}.log 2>&1
    if ! grep -q "STUDENT_${ARM}_DONE" /workspace/vla/ft_logs/train_student_${ARM}.log 2>/dev/null; then
      stamp "$ARM: train FAILED -> ft_logs/train_student_${ARM}.log"; tail -6 /workspace/vla/ft_logs/train_student_${ARM}.log; exit 1
    fi
  fi
  # 2. assemble + per-step retention (KEEP ckpt)
  stamp "$ARM: per-step retention"
  ARM=$ARM SAVE_ROOT=$SAVE N_KEYS=${NK[$ARM]} PERSTEP=1 GPU=4 N_EPS=40 \
    OUT_JSON=$LB/retention_perstep_${ARM}.json \
    bash $LB/assemble_and_retn.sh
  if [ ! -f "$LB/retention_perstep_${ARM}.json" ]; then stamp "$ARM: no perstep json"; exit 1; fi
  stamp "=== $ARM DONE (ckpt KEPT at $SAVE) ==="
done

stamp "ALL DONE"
echo "=== LingBot PER-STEP entropy curve (retention) ==="
for ARM in dc hashmod40 hashmod160 hash; do
  echo "  $ARM: $(/usr/bin/python3.11 -c "import json;d=json.load(open('$LB/retention_perstep_${ARM}.json'));print('retn',round(d['retention'],3),'survives',d['survives'],'Z',round(d['z'],2))")"
done
echo "LINGBOT_PERSTEP_COMPLETE"
