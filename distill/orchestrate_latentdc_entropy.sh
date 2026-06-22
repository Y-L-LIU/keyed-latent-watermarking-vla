#!/usr/bin/env bash
# Entropy-sweep arm of the latent-DC distillation test (ONE value of N_KEYS).
#   NKEYS=160 GPUS="2 3 4 5" bash orchestrate_latentdc_entropy.sh
# relabel (obs-bucket-mod-N DC seed, sharded) -> train LoRA student (params-only)
# -> roll out plain (MAP recovers seed, sharded) -> obs-keyed latent-DC detector vs
# the existing clean-control student. Deletes the bulky ckpt + relabel data when done.
set -uo pipefail
DISTILL=/workspace/vla/distill; LOGD=$DISTILL/logs; mkdir -p "$LOGD"
NKEYS=${NKEYS:?set NKEYS}
read -r -a GPU <<< "${GPUS:-2 3 4 5}"
NG=${#GPU[@]}
DATA=$DISTILL/data/libero_goal_latentdc_n${NKEYS}_k42
CFG=pi05_libero_goal_lora_distill_latentdc_n${NKEYS}
EXP=distill_latentdc_n${NKEYS}_k42
CKPT=/workspace/vla/openpi-checkpoints/$CFG/$EXP/1499
TAG=latentdc_n${NKEYS}_student
OUT=$DISTILL/eval
stamp(){ echo "[$(date +%H:%M:%S)] N=$NKEYS $*"; }
export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi:/workspace/vla/openpi/packages/openpi-client/src:/workspace/vla/openpi/third_party/libero:/workspace/vla/distill
export OPENPI_DATA_HOME=/workspace/vla/openpi-cache JAX_COMPILATION_CACHE_DIR=/workspace/vla/jax_cache WANDB_MODE=disabled
PY=/usr/bin/python3.11

# task shards across the available GPUs (10 libero_goal tasks)
OFFS=(0 3 6 8); CNTS=(3 3 2 2)   # 4-way; trimmed below if fewer GPUs

# ---- 1. relabel (sharded) ----
stamp "relabel start (obs-bucket DC, N_KEYS=$NKEYS) -> $DATA"
rm -f $LOGD/relabel_ldc_n${NKEYS}_shard*.log
pids=()
for i in $(seq 0 $((NG-1))); do
  lo=${OFFS[$i]}; hi=$((lo + CNTS[$i]))
  CUDA_VISIBLE_DEVICES=${GPU[$i]} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    nohup $PY $DISTILL/relabel_latent_dc_obskey.py --out-dir "$DATA" --n-keys "$NKEYS" \
      --secret-key 42 --beta 1.0 --task-range "$lo" "$hi" \
      > "$LOGD/relabel_ldc_n${NKEYS}_shard${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
ndone=$(grep -l "ldc-obskey] DONE" $LOGD/relabel_ldc_n${NKEYS}_shard*.log 2>/dev/null | wc -l)
if [ "$ndone" -ne "$NG" ]; then stamp "relabel FAILED ($ndone/$NG shards) -> $LOGD/relabel_ldc_n${NKEYS}_shard*.log"; exit 1; fi
stamp "relabel done ($(ls $DATA/libero_goal/*.hdf5 | wc -l) tasks)"

# ---- 2. train student (params-only) ----
GPUSCSV=$(IFS=,; echo "${GPU[*]}")
stamp "train student start (gpus=$GPUSCSV)"
OPENPI_PARAMS_ONLY=1 CFG=$CFG EXP=$EXP GPUS=$GPUSCSV bash $DISTILL/run_train_student.sh > $LOGD/launch_ldc_n${NKEYS}.log 2>&1
if [ ! -d "$CKPT/params" ]; then stamp "train FAILED, no $CKPT/params -> $LOGD/train_$CFG.log"; tail -5 $LOGD/train_$CFG.log; exit 1; fi
stamp "train done -> $CKPT"
rm -rf "$DATA"   # relabel corpus no longer needed (norm-stats + train consumed it)
stamp "deleted relabel data $DATA"

# ---- 3. roll out student plain (MAP recovers seed), sharded ----
stamp "rollout start (sharded $NG-way)"
pids=()
for i in $(seq 0 $((NG-1))); do
  OUT=$OUT GPU=${GPU[$i]} TAG=$TAG TASKS=${CNTS[$i]} OFFSET=${OFFS[$i]} TRIALS=5 \
    POLICY_CFG=$CFG POLICY_CKPT=$CKPT \
    DET_CFG=pi05_libero DET_CKPT=/workspace/vla/models/pi05_libero KEYING=observation \
    nohup bash $DISTILL/run_eval_obstied.sh > "$LOGD/eval_ldc_n${NKEYS}_shard${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
nroll=$(ls $OUT/libero_goal_${TAG}/rollouts/task_rollout/*_plain.npz 2>/dev/null | wc -l)
stamp "rollout done ($nroll episodes)"

# ---- 4. score (obs-keyed DC detector) vs clean control ----
stamp "scoring"
$PY $DISTILL/score_latentdc_obskey.py --n-keys "$NKEYS" \
  --latentdc-rollouts $OUT/libero_goal_${TAG}/rollouts/task_rollout \
  --clean-rollouts   $OUT/libero_goal_clean_student/rollouts/task_rollout \
  --secret-key 42 2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprec|flax" | tee $DISTILL/VERDICT_latentdc_n${NKEYS}.txt

# ---- 5. free the checkpoint ----
rm -rf "/workspace/vla/openpi-checkpoints/$CFG"
stamp "DONE (ckpt deleted) -> $DISTILL/VERDICT_latentdc_n${NKEYS}.txt"
echo "ORCH_LDC_N${NKEYS}_COMPLETE"
