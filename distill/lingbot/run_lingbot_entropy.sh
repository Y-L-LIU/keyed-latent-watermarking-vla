#!/usr/bin/env bash
# LingBot entropy sweep to mirror pi0.5: obs-tied key (same gaussian vector function as
# the hash arm) with the bucket folded mod N_KEYS -- a pure cardinality knob. Reuses the
# per-task DC (~10, retention 0.48) and full hash (~326, 0.21) as the curve endpoints;
# this adds the interior points N=40, 160. Sequential (one 48G ckpt at a time).
set -uo pipefail
LB=/workspace/vla/distill/lingbot
stamp(){ echo "[$(date +%H:%M:%S)] LB-ENT $*"; }
export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/distill:${PYTHONPATH:-}

for N in 40 160; do
  ARM=hashmod${N}
  DS=/workspace/vla/lingbot_latents/relabel_${ARM}
  SAVE=/workspace/vla/lingbot_out/distill_ckpt/student_${ARM}
  stamp "=== N=$N start ==="
  # 1. relabel if missing
  if [ "$(ls $DS/data/chunk-000/*.parquet 2>/dev/null | wc -l)" -lt 500 ]; then
    stamp "relabel N=$N"
    /usr/bin/python3.11 $LB/relabel.py --arm hashmod --n-keys $N --out $DS \
      > /workspace/vla/ft_logs/relabel_${ARM}.log 2>&1
    grep -q "DONE ->" /workspace/vla/ft_logs/relabel_${ARM}.log || { stamp "relabel N=$N FAILED"; exit 1; }
  fi
  stamp "relabel ok ($(ls $DS/data/chunk-000/*.parquet | wc -l) parquet)"
  # 2. train student (gpu4-7)
  stamp "train N=$N"
  ARM=$ARM NUM_STEPS=1500 bash $LB/launch_train.sh > /workspace/vla/ft_logs/launch_${ARM}.log 2>&1
  if ! grep -q "STUDENT_${ARM}_DONE" /workspace/vla/ft_logs/train_student_${ARM}.log 2>/dev/null; then
    stamp "train N=$N FAILED -> ft_logs/train_student_${ARM}.log"; tail -6 /workspace/vla/ft_logs/train_student_${ARM}.log; exit 1
  fi
  # 3. assemble + retention (gpu2), pass N_KEYS
  stamp "retention N=$N"
  ARM=$ARM N_KEYS=$N GPU=2 N_EPS=40 bash $LB/assemble_and_retn.sh
  if [ ! -f "$LB/retention_${ARM}.json" ]; then stamp "retention N=$N: no json"; exit 1; fi
  # 4. free the 48G ckpt
  rm -rf "$SAVE" /workspace/vla/lingbot_out/distill_ckpt/assembled_${ARM}
  stamp "=== N=$N DONE (ckpt freed) -> retention_${ARM}.json ==="
done

stamp "ALL DONE"
echo "=== LingBot entropy curve (retention) ==="
echo "  per-task DC (~10): $(/usr/bin/python3.11 -c "import json;print(round(json.load(open('$LB/retention_dc.json'))['retention'],3))")"
for N in 40 160; do
  echo "  hashmod N=$N: $(/usr/bin/python3.11 -c "import json;print(round(json.load(open('$LB/retention_hashmod${N}.json'))['retention'],3))")"
done
echo "  hash full (~326): $(/usr/bin/python3.11 -c "import json;print(round(json.load(open('$LB/retention_hash.json'))['retention'],3))")"
echo "LINGBOT_ENTROPY_COMPLETE"
