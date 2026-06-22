"""Phase B cost-utility-style aggregator across all sweep tags.

Reuses _per_episode_z + _wmf_score helpers from build_cost_utility_table.py.
Walks /workspace/scratch/anon/libero10_wm_postprocess_full/<tag>/<tag>/task_rollout/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_cost_utility_table import _per_episode_z


def process_tag(rollout_dir: pathlib.Path, *, null_count, max_windows,
                score_step_scope, subspace_rank, group_sizes):
    if not rollout_dir.exists():
        return {"tag": rollout_dir.parts[-3], "error": f"no dir {rollout_dir}"}
    plain_z, wm_z, plain_succ, wm_succ = [], [], [], []
    for npz in sorted(rollout_dir.glob("*.npz")):
        data = np.load(npz, allow_pickle=True)
        variant = str(data["variant"])
        success = bool(data["success"]) if "success" in data.files else False
        wmf, z = _per_episode_z(
            data, null_count=null_count, max_windows=max_windows,
            score_step_scope=score_step_scope, subspace_rank=subspace_rank,
        )
        if variant == "plain":
            plain_z.append(z); plain_succ.append(success)
        else:
            wm_z.append(z); wm_succ.append(success)

    p = np.asarray(plain_z); w = np.asarray(wm_z)
    single_auc = float(sum(1 for pi in p for wi in w if wi > pi) / max(len(p) * len(w), 1))

    rng = np.random.default_rng(42)
    group_aucs: dict[str, float] = {}
    for gs in group_sizes:
        if gs > len(p) or gs > len(w):
            group_aucs[f"cross_at_{gs}_auc"] = float("nan")
            continue
        pg, wg = [], []
        for _ in range(1000):
            pg.append(p[rng.choice(len(p), size=gs, replace=False)].sum())
            wg.append(w[rng.choice(len(w), size=gs, replace=False)].sum())
        pg = np.asarray(pg); wg = np.asarray(wg)
        group_aucs[f"cross_at_{gs}_auc"] = float(
            sum(1 for pi in pg for wi in wg if wi > pi) / (len(pg) * len(wg))
        )

    return {
        "n_plain": int(len(p)), "n_wm": int(len(w)),
        "task_success_plain": float(np.mean(plain_succ)) if plain_succ else float("nan"),
        "task_success_wm": float(np.mean(wm_succ)) if wm_succ else float("nan"),
        "wm_z_mean": float(np.mean(w)) if len(w) else float("nan"),
        "plain_z_mean": float(np.mean(p)) if len(p) else float("nan"),
        "wm_minus_plain_mean": float(np.mean(w) - np.mean(p)) if len(w) and len(p) else float("nan"),
        "single_auc": single_auc,
        **group_aucs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/scratch/anon/libero10_wm_postprocess_full")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--out", default="/workspace/vla/attack_c_data/cost_utility_table_phaseB_sweep.json")
    parser.add_argument("--null-count", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--score-step-scope", default="full_chunk")
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    rows = []
    for tag in args.tags:
        print(f"[phaseB] processing {tag} ...", file=sys.stderr, flush=True)
        rollout_dir = root / tag / tag / "task_rollout"
        row = {"tag": tag, **process_tag(
            rollout_dir,
            null_count=args.null_count, max_windows=args.max_windows,
            score_step_scope=args.score_step_scope, subspace_rank=args.subspace_rank,
            group_sizes=tuple(args.group_sizes),
        )}
        print(json.dumps(row), file=sys.stderr, flush=True)
        rows.append(row)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2))

    print("\n| tag | n_wm | n_pl | succ_wm | succ_pl | single AUC | cross@5 | cross@10 | cross@15 | Δ z |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            print(f"| {r['tag']} | ERROR | | | | | | | | |"); continue
        print("| {tag} | {n_wm} | {n_plain} | {sw:.2f} | {sp:.2f} | {sa:.3f} | {c5:.3f} | {c10:.3f} | {c15:.3f} | {dz:.2f} |".format(
            tag=r['tag'], n_wm=r['n_wm'], n_plain=r['n_plain'],
            sw=r['task_success_wm'], sp=r['task_success_plain'],
            sa=r['single_auc'], c5=r['cross_at_5_auc'],
            c10=r['cross_at_10_auc'], c15=r['cross_at_15_auc'],
            dz=r['wm_minus_plain_mean']))


if __name__ == "__main__":
    main()
