#!/usr/bin/env bash
# Single entrypoint. Usage:
#   bash run.sh list                 # show every section -> wrapper
#   bash run.sh check                # preflight (what's runnable now)
#   bash run.sh all-analysis         # regenerate all CPU-only paper assets
#   bash run.sh <name>               # run one analysis section, e.g.  run.sh verification
#   bash run.sh train  <openpi|lingbot> <libero|robotwin>  # GPU LoRA descendant fine-tune
#   bash run.sh rollout <wrapper> [args]                   # GPU rollout, e.g. rollout openpi_verification
# names: spectrum bandstop latent-bandstop recovery verification identification
#        uniqueness unforgeability
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; A="$HERE/analysis"; T="$HERE/train"; R="$HERE/rollout"
declare -A MAP=(
  [spectrum]="sec5.1_spectrum"            [bandstop]="sec5.1_bandstop"
  [latent-bandstop]="sec5.1_latent_bandstop" [recovery]="sec5.2_6.2_recovery"
  [verification]="sec6.3_verification"    [identification]="sec6.5_identification"
  [uniqueness]="sec7.1_uniqueness"        [unforgeability]="sec7.2_unforgeability"
)
case "${1:-list}" in
  list)
    printf '%-16s %-34s %s\n' NAME WRAPPER "PAPER §"
    printf '%-16s %-34s %s\n' spectrum        analysis/sec5.1_spectrum.sh        "5.1 fig_spectrum"
    printf '%-16s %-34s %s\n' bandstop        analysis/sec5.1_bandstop.sh        "5.1 fig_bandstop_sweep"
    printf '%-16s %-34s %s\n' latent-bandstop analysis/sec5.1_latent_bandstop.sh "5.1 (GPU) latent curve"
    printf '%-16s %-34s %s\n' recovery        analysis/sec5.2_6.2_recovery.sh    "5.2/6.2 tab_recovery"
    printf '%-16s %-34s %s\n' verification    analysis/sec6.3_verification.sh    "6.3 tab_main/utility+figs"
    printf '%-16s %-34s %s\n' identification  analysis/sec6.5_identification.sh  "6.5 tab_identification"
    printf '%-16s %-34s %s\n' uniqueness      analysis/sec7.1_uniqueness.sh      "7.1 key_collision"
    printf '%-16s %-34s %s\n' unforgeability  analysis/sec7.2_unforgeability.sh  "7.2 unforgeability"
    echo
    echo "train (GPU LoRA descendants, feeds tab_descendant):"
    printf '  %-44s %s\n' "train/openpi_lora.sh  [libero|robotwin]" "pi0.5 descendant"
    printf '  %-44s %s\n' "train/lingbot_lora.sh [libero|robotwin]" "lingbot descendant"
    printf '  %-44s %s\n' "train/build_compressed_ckpt.sh ..."      "§12.5 prune/quant model surgery"
    echo
    echo "rollout (GPU) wrappers live in rollout/ — see README.md"
    ;;
  check)        exec bash "$HERE/check.sh" ;;
  all-analysis) exec bash "$A/run_all.sh" ;;
  train)
    shift; fam="${1:?usage: run.sh train <openpi|lingbot> <libero|robotwin>}"; shift || true
    case "$fam" in
      openpi)  exec bash "$T/openpi_lora.sh" "$@" ;;
      lingbot) exec bash "$T/lingbot_lora.sh" "$@" ;;
      *) echo "run.sh train <openpi|lingbot> <libero|robotwin>"; exit 2 ;;
    esac ;;
  rollout)
    shift; w="${1:?usage: run.sh rollout <wrapper> [args]}"; shift || true
    [ -f "$R/$w.sh" ] || { echo "no such rollout wrapper: $w"; ls "$R" | sed 's/\.sh$//'; exit 2; }
    exec bash "$R/$w.sh" "$@" ;;
  *)
    w="${MAP[$1]:-}"; [ -z "$w" ] && { echo "unknown: $1"; bash "$0" list; exit 2; }
    shift; exec bash "$A/$w.sh" "$@" ;;
esac
