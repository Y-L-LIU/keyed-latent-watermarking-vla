#!/usr/bin/env bash
# LingBot high-entropy NEGATIVE arm (completes "same pattern holds on LingBot-VA"):
# train a hash-keyed (high-entropy, obs-tied) student on the staged relabel_hash corpus,
# then assemble + retention. Expected: does NOT survive (mirrors openpi deployed key),
# vs the DC positive control (retention_dc.json survives).
set -uo pipefail
LB=/workspace/vla/distill/lingbot
stamp(){ echo "[$(date +%H:%M:%S)] HASH-NEG $*"; }
stamp "train start (relabel_hash, gpu4-7)"
ARM=hash NUM_STEPS=1500 bash $LB/launch_train.sh
if ! grep -q "STUDENT_hash_DONE" /workspace/vla/ft_logs/train_student_hash.log 2>/dev/null; then
  stamp "train marker missing -> /workspace/vla/ft_logs/train_student_hash.log"; tail -8 /workspace/vla/ft_logs/train_student_hash.log; exit 1
fi
stamp "train done; assemble + retention (gpu2)"
ARM=hash GPU=2 N_EPS=40 bash $LB/assemble_and_retn.sh
stamp "DONE -> $LB/retention_hash.json"
echo "=== LingBot arms ==="
for a in dc hash; do echo "--$a--"; cat $LB/retention_${a}.json 2>/dev/null; echo; done
echo "LINGBOT_HASH_NEG_COMPLETE"
