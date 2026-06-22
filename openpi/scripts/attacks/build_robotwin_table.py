"""Aggregator: walk the robotwin sweep output, compute WMF z-scores, and emit
a cost-utility table (one row per attack condition).

Output layout it walks:
    /workspace/scratch/anon/robotwin2/wm_eval/<TAG>/<TASK>/pi05_aloha_full_base/4000/episode_NNN_{plain,watermarked}.npz

Reuses _per_episode_z from build_cost_utility_table.py. RoboTwin's eval script
uses secret_key=17 (default) and sample_rate_hz=50 — neither is saved into the
npz, so we override both at scoring time.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_cost_utility_table as bcu  # noqa: E402


def _patched_per_episode_z(data, *, null_count, max_windows, score_step_scope,
                           subspace_rank, secret_key, sample_rate_hz):
    """Drop-in for bcu._per_episode_z that lets us inject robotwin's secret_key/fs."""
    true_vec = bcu._score_episode(data, score_step_scope=score_step_scope, max_windows=max_windows)
    null_rows = np.stack([
        bcu._null_episode(
            data, off + 1, score_step_scope=score_step_scope, max_windows=max_windows,
            secret_key=secret_key, sample_rate_hz=sample_rate_hz,
        )
        for off in range(null_count)
    ])
    wmf = bcu._wmf_score(true_vec, null_rows, subspace_rank=subspace_rank)
    null_wmfs = np.array([
        bcu._wmf_score(null_rows[i], np.delete(null_rows, i, axis=0), subspace_rank=subspace_rank)
        for i in range(null_rows.shape[0])
    ], dtype=np.float64)
    null_std = float(np.std(null_wmfs))
    if null_std < 1e-6:
        null_std = 1.0
    z = (wmf - float(np.mean(null_wmfs))) / null_std
    return float(wmf), float(z)


def process_tag(tag_dir: pathlib.Path, *, null_count, max_windows,
                score_step_scope, subspace_rank, group_sizes,
                secret_key, sample_rate_hz):
    if not tag_dir.exists():
        return {"tag": tag_dir.name, "error": f"no dir {tag_dir}"}
    plain_z, wm_z, plain_succ, wm_succ = [], [], [], []
    npz_paths = sorted(tag_dir.glob("*/pi05_aloha_full_base/*/*.npz"))
    for npz in npz_paths:
        data = np.load(npz, allow_pickle=True)
        variant = str(data["variant"])
        success = bool(data["success"]) if "success" in data.files else False
        try:
            _wmf, z = _patched_per_episode_z(
                data, null_count=null_count, max_windows=max_windows,
                score_step_scope=score_step_scope, subspace_rank=subspace_rank,
                secret_key=secret_key, sample_rate_hz=sample_rate_hz,
            )
        except Exception as exc:
            print(f"[skip] {npz}: {exc}", file=sys.stderr)
            continue
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
    parser.add_argument("--root", default="/workspace/scratch/anon/robotwin2/wm_eval")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--out", default="/workspace/vla/attack_c_data/cost_utility_table_robotwin_sweep.json")
    parser.add_argument("--null-count", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--score-step-scope", default="full_chunk")
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    parser.add_argument("--secret-key", type=int, default=17)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    rows = []
    for tag in args.tags:
        print(f"[robotwin] processing {tag} ...", file=sys.stderr, flush=True)
        row = {"tag": tag, **process_tag(
            root / tag,
            null_count=args.null_count, max_windows=args.max_windows,
            score_step_scope=args.score_step_scope, subspace_rank=args.subspace_rank,
            group_sizes=tuple(args.group_sizes),
            secret_key=args.secret_key, sample_rate_hz=args.sample_rate_hz,
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
            sa=r['single_auc'],
            c5=r.get('cross_at_5_auc', float('nan')),
            c10=r.get('cross_at_10_auc', float('nan')),
            c15=r.get('cross_at_15_auc', float('nan')),
            dz=r['wm_minus_plain_mean']))


if __name__ == "__main__":
    main()
