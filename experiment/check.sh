#!/usr/bin/env bash
# Preflight: from inside experiment/, report what's runnable NOW.
# Checks the env, each analysis target, and whether its input data is present.
set -uo pipefail; source "$(dirname "$0")/env.sh"
ok(){ printf '  \033[32mok\033[0m   %s\n' "$1"; }
no(){ printf '  \033[31mMISS\033[0m %s\n' "$1"; }
cnt(){ ls $1 2>/dev/null | wc -l | tr -d ' '; }

echo "env:"
[ -x "$PY" ] && ok "python $PY" || no "python $PY"
echo
echo "inputs (rollout data the CPU analysis reads):"
n=$(cnt "$VLA/attack_c_data/per_episode_scores/*_partial_map_*.csv")
[ "$n" -gt 0 ] && ok "per_episode_scores: $n csv  (§6.3 verification, §6.5 identification)" || no "per_episode_scores csv (run rollout/export_scores)"
n=$(cnt "$VLA/eval_out/base/libero_10/rollouts/none/task_rollout/*_watermarked.npz")
[ "$n" -gt 0 ] && ok "clean pi05 wm npz: $n  (§7.1, §7.2, §5.2/6.2 recovery, §5.1 latent)" || no "clean pi05 wm npz (run rollout/openpi_verification)"
n=$(cnt "$VLA/eval_out/lingbot_libero10_descendant/libero_10/*.npz")
[ "$n" -gt 0 ] && ok "lingbot clean npz: $n  (§5.1 spectrum/bandstop)" || no "lingbot clean npz (run rollout/lingbot_verification)"
n=$(cnt "$VLA/attack_c_data/per_episode_scores_descendant/*_partial_map_*.csv")
[ "$n" -gt 0 ] && ok "descendant scores: $n  (§6.5 weight-level rows)" || no "descendant scores (optional; §6.5 weight-level rows skipped)"
echo
echo "analysis targets compile:"
for f in make_fig_spectrum make_fig_bandstop_sweep analyze_verification analyze_identification \
         make_key_collision_analysis make_unforgeability_analysis; do
  "$PY" -m py_compile "$VLA/results/$f.py" 2>/dev/null && ok "$f.py" || no "$f.py"
done
echo
echo "-> CPU assets:  bash run.sh all-analysis     (or run.sh list)"
echo "-> GPU rollouts: see rollout/  and README.md (most data already exists)"
