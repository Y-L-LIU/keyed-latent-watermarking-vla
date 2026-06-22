#!/usr/bin/env bash
# Driver: latent-DC entropy sweep, two arms SEQUENTIAL (low disk peak), 4 GPUs each.
# high (N=160, ~deployed cardinality) first -- the decisive corner -- then medium (N=40).
set -uo pipefail
DISTILL=/workspace/vla/distill
stamp(){ echo "[$(date +%H:%M:%S)] DRIVER $*"; }
for N in 160 40; do
  stamp "=== arm N_KEYS=$N start ==="
  NKEYS=$N GPUS="2 3 4 5" bash $DISTILL/orchestrate_latentdc_entropy.sh
  rc=$?
  stamp "=== arm N_KEYS=$N exit=$rc ==="
  [ $rc -ne 0 ] && { stamp "ABORT (arm $N failed)"; exit $rc; }
done
stamp "ALL ARMS DONE"
echo "=== entropy sweep summary ==="
echo "low  (per-task, ~10 keys):  AUC 0.89   [existing VERDICT_latentdc.txt]"
for N in 40 160; do
  a=$(grep -oE "AUC \(latentdc-N${N} vs clean\) = [0-9.]+" $DISTILL/VERDICT_latentdc_n${N}.txt 2>/dev/null | grep -oE "[0-9.]+$")
  echo "N=${N} obs-bucket DC: AUC ${a:-?}   [VERDICT_latentdc_n${N}.txt]"
done
echo "ENTROPY_SWEEP_COMPLETE"
