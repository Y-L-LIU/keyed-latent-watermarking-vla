#!/usr/bin/env bash
# Autonomous: wait for the output-arm student, roll it out plain (sharded GPUs 0-3),
# then score the OUTPUT detector (executed action vs obs-tied r_out) and compare to
# the clean-control student (re-scored with the same output detector). Reports the
# distillation-survival verdict for the perceptible arm + SR cost.
set -uo pipefail
DISTILL=/workspace/vla/distill; LOGD=$DISTILL/logs
CKPT=/workspace/vla/openpi-checkpoints/pi05_libero_goal_lora_distill_output/distill_output_k42/1499
stamp(){ echo "[$(date +%H:%M:%S)] $*"; }

stamp "ORCH-OUT: waiting for output-student checkpoint..."
until [ -d "$CKPT/params" ]; do sleep 30; done
sleep 10
stamp "ORCH-OUT: checkpoint present; launching plain rollouts (GPUs 0-3)"

OFFS=(0 3 6 8); CNTS=(3 3 2 2)
for s in 0 1 2 3; do
  OUT=$DISTILL/eval GPU=$s TAG=output_student TASKS=${CNTS[$s]} OFFSET=${OFFS[$s]} TRIALS=5 \
    POLICY_CFG=pi05_libero_goal_lora_distill_output POLICY_CKPT=$CKPT \
    DET_CFG=pi05_libero DET_CKPT=/workspace/vla/models/pi05_libero \
    SECRET_KEY=42 Q=0.08 PROJ=0,1,2 KEYING=observation \
    nohup bash $DISTILL/run_eval_obstied.sh > "$LOGD/eval_output_student_shard${s}.log" 2>&1 &
done
wait
stamp "ORCH-OUT: rollouts done; scoring OUTPUT detector"

export PYTHONPATH=/workspace/vla/openpi/src:/workspace/vla/openpi/packages/openpi-client/src
/usr/bin/python3.11 $DISTILL/analyze_distill.py \
  --obstied-rollouts $DISTILL/eval/libero_goal_output_student/rollouts/task_rollout \
  --clean-rollouts   $DISTILL/eval/libero_goal_clean_student/rollouts/task_rollout \
  --secret-key 42 --q 0.08 --proj-dims 0,1,2 --signal-key chunk_observed_actions \
  2>&1 | grep -vE "WARNING|warn|tcmalloc|Deprecation|flax" | tee $DISTILL/VERDICT_output.txt

# SR cost (task success) for output vs clean students
/usr/bin/python3.11 - <<'EOF' 2>&1 | tee -a $DISTILL/VERDICT_output.txt
import glob, numpy as np
def sr(d):
    s=[]
    for p in glob.glob(d+"/*_plain.npz"):
        try: s.append(float(np.load(p)["success"]))
        except Exception: pass
    return (np.mean(s), len(s)) if s else (float("nan"),0)
for tag,d in [("output student","/workspace/vla/distill/eval/libero_goal_output_student/rollouts/task_rollout"),
              ("clean  student","/workspace/vla/distill/eval/libero_goal_clean_student/rollouts/task_rollout")]:
    m,n=sr(d); print(f"SR {tag}: {m:.2f} (n={n})")
EOF
stamp "ORCH-OUT: DONE -> $DISTILL/VERDICT_output.txt"
echo "ORCH_OUTPUT_COMPLETE"
