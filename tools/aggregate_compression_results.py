#!/usr/bin/env python3
"""Aggregate prune/quant robustness eval outputs into a single markdown table.

Reads:
- openpi LIBERO:  eval_out_compression/libero_{suite}_{attack}/reports/none/summary_task_rollout.json
- openpi RoboTwin: eval_out_compression/openpi_robotwin_{attack}/seed_{N}/.../summary.json
- lingbot LIBERO:  eval_out_compression/lingbot_libero10_{attack}_{wm|plain}/libero_10/*.npz
- lingbot RoboTwin: eval_out_compression/lingbot_robotwin_{attack}_{wm|plain}/seedNNN/.../<task>/*.npz

For openpi LIBERO the eval already computes AUC/TPR. For the rest we re-derive AUC
from the WMF / score fields in the saved NPZ.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------------------------
def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort() + 1
    pos_rank_sum = ranks[: pos.size].sum()
    return float((pos_rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, target_fpr: float) -> float:
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    k = int(np.ceil(target_fpr * neg.size))
    if k <= 0:
        return 0.0
    thresh = np.sort(neg)[-k]
    return float(np.mean(pos > thresh))


def group_auc(pos: np.ndarray, neg: np.ndarray, G: int = 16, n: int = 4000, seed: int = 0) -> float:
    """AUC at group budget |G| (mean-aggregate G episodes per decision), MC-estimated.
    Matches the |G|=16 column reported in the §6 main table so compression is same-unit."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pg = np.array([pos[rng.integers(0, pos.size, G)].mean() for _ in range(n)])
    ng = np.array([neg[rng.integers(0, neg.size, G)].mean() for _ in range(n)])
    return roc_auc(pg, ng)


# -----------------------------------------------------------------------------
def openpi_libero(suite: str, attack: str, base: pathlib.Path, suffix: str = "") -> dict:
    summary = base / f"libero_{suite}_{attack}{suffix}" / "reports" / "none" / "summary_task_rollout.json"
    if not summary.exists():
        return {"attack": attack, "status": "missing"}
    s = json.loads(summary.read_text())
    rollout_dir = base / f"libero_{suite}_{attack}{suffix}" / "rollouts" / "none" / "task_rollout"
    succ_wm = succ_plain = 0
    n_wm = n_plain = 0
    pos_scores, neg_scores = [], []
    for f in rollout_dir.glob("*.npz"):
        try:
            d = np.load(f, allow_pickle=True)
            is_wm = "wm" in f.name or str(d["variant"]) == "watermarked"
            success = bool(d["success"])
            sc = (float(d["score"]) if "score" in d.files
                  else float(np.sum(d["wmf_scores"])) if "wmf_scores" in d.files else None)
            if is_wm:
                n_wm += 1; succ_wm += int(success)
                if sc is not None: pos_scores.append(sc)
            else:
                n_plain += 1; succ_plain += int(success)
                if sc is not None: neg_scores.append(sc)
        except Exception:
            continue
    pos = np.asarray(pos_scores); neg = np.asarray(neg_scores)
    return {
        "attack": attack,
        "auc": s.get("roc_auc"),
        "auc16": group_auc(pos, neg) if pos.size and neg.size else None,
        "tpr_1": s.get("tpr_at_1pct_fpr"),
        "tpr_10": s.get("tpr_at_10pct_fpr"),
        "wm_count": int(s.get("episode_count_watermarked", n_wm)),
        "plain_count": int(s.get("episode_count_plain", n_plain)),
        "success_wm": f"{succ_wm}/{n_wm}" if n_wm else "?",
        "success_plain": f"{succ_plain}/{n_plain}" if n_plain else "?",
        "wm_minus_plain_mean": s.get("wm_minus_plain_mean"),
    }


def openpi_robotwin(attack: str, base: pathlib.Path, suffix: str = "") -> dict:
    # New layout: openpi_robotwin10_<attack>/<task>/<config>/<ckpt>/episode_*_{plain,watermarked}.npz
    # (10 robotwin10_clean tasks pooled; bbh was the wrong task — not in training set).
    root = base / f"openpi_robotwin10_{attack}{suffix}"
    if not root.exists():
        return {"attack": attack, "status": "missing"}
    pos_scores, neg_scores = [], []
    succ_wm = succ_plain = n_wm = n_plain = 0
    for npz in root.rglob("episode_*_watermarked.npz"):
        d = np.load(npz, allow_pickle=True)
        pos_scores.append(float(d["score"]))
        n_wm += 1; succ_wm += int(bool(d["success"]))
    for npz in root.rglob("episode_*_plain.npz"):
        d = np.load(npz, allow_pickle=True)
        neg_scores.append(float(d["score"]))
        n_plain += 1; succ_plain += int(bool(d["success"]))
    pos = np.asarray(pos_scores); neg = np.asarray(neg_scores)
    return {
        "attack": attack,
        "auc": roc_auc(pos, neg) if pos.size and neg.size else None,
        "auc16": group_auc(pos, neg) if pos.size and neg.size else None,
        "tpr_1": tpr_at_fpr(pos, neg, 0.01) if pos.size and neg.size else None,
        "tpr_10": tpr_at_fpr(pos, neg, 0.10) if pos.size and neg.size else None,
        "wm_count": int(n_wm),
        "plain_count": int(n_plain),
        "success_wm": f"{succ_wm}/{n_wm}",
        "success_plain": f"{succ_plain}/{n_plain}",
        "wm_score_mean": float(pos.mean()) if pos.size else None,
        "plain_score_mean": float(neg.mean()) if neg.size else None,
    }


def lingbot_libero(attack: str, base: pathlib.Path) -> dict:
    wm_dir = base / f"lingbot_libero10_{attack}_wm" / "libero_10"
    pl_dir = base / f"lingbot_libero10_{attack}_plain" / "libero_10"
    # Per-episode WMF and z-score aggregation:
    # Each NPZ has chunk_wmf_scores (per-chunk) and chunk_map_mse.
    def collect(root):
        wmf_per_ep = []
        success_n = total_n = 0
        for npz in root.rglob("task*_ep*.npz"):
            try:
                d = np.load(npz, allow_pickle=True)
            except Exception:
                continue
            wmfs = d["wmf_scores"] if "wmf_scores" in d.files else np.zeros(0)
            wmf_per_ep.append(float(np.sum(wmfs)) if wmfs.size else 0.0)
            total_n += 1
            success_n += int(bool(d["success"])) if "success" in d.files else 0
        return np.asarray(wmf_per_ep), success_n, total_n

    pos, sw, nw = collect(wm_dir) if wm_dir.exists() else (np.zeros(0), 0, 0)
    neg, sp, np_ = collect(pl_dir) if pl_dir.exists() else (np.zeros(0), 0, 0)
    return {
        "attack": attack,
        "auc": roc_auc(pos, neg) if pos.size and neg.size else None,
        "auc16": group_auc(pos, neg) if pos.size and neg.size else None,
        "tpr_1": tpr_at_fpr(pos, neg, 0.01) if pos.size and neg.size else None,
        "tpr_10": tpr_at_fpr(pos, neg, 0.10) if pos.size and neg.size else None,
        "wm_count": int(nw), "plain_count": int(np_),
        "success_wm": f"{sw}/{nw}", "success_plain": f"{sp}/{np_}",
        "wm_score_mean": float(pos.mean()) if pos.size else None,
        "plain_score_mean": float(neg.mean()) if neg.size else None,
    }


def lingbot_robotwin(attack: str, base: pathlib.Path) -> dict:
    # Corrected layout: lingbot_robotwin10_<attack>_<wm|plain>/<task>/task*_ep*.npz
    # (inline-map2 worktree, 10 tasks, --chunk-period 2 --map-num-starts 4; the old
    # lingbot_robotwin_* with main-branch MAP + period 6 gave a spurious AUC~0.5).
    # Prefer the corrected ACTION_SNR_SHIFT=0.05 re-run (the default-suffix dirs were run at the
    # broken RoboTwin-native shift=1.0 -> contaminated AUC).
    wm_dir = base / f"lingbot_robotwin10_{attack}_wm_snr05"
    pl_dir = base / f"lingbot_robotwin10_{attack}_plain_snr05"
    if not wm_dir.exists():
        wm_dir = base / f"lingbot_robotwin10_{attack}_wm"
        pl_dir = base / f"lingbot_robotwin10_{attack}_plain"

    def collect(root):
        wmf_per_ep = []
        success_n = total_n = 0
        if root.exists():
            for npz in root.rglob("*.npz"):
                try:
                    d = np.load(npz, allow_pickle=True)
                except Exception:
                    continue
                wmfs = d["wmf_scores"] if "wmf_scores" in d.files else np.zeros(0)
                wmf_per_ep.append(float(np.sum(wmfs)) if wmfs.size else 0.0)
                total_n += 1
                success_n += int(bool(d["success"])) if "success" in d.files else 0
        return np.asarray(wmf_per_ep), success_n, total_n

    pos, sw, nw = collect(wm_dir)
    neg, sp, np_ = collect(pl_dir)
    return {
        "attack": attack,
        "auc": roc_auc(pos, neg) if pos.size and neg.size else None,
        "auc16": group_auc(pos, neg) if pos.size and neg.size else None,
        "tpr_1": tpr_at_fpr(pos, neg, 0.01) if pos.size and neg.size else None,
        "tpr_10": tpr_at_fpr(pos, neg, 0.10) if pos.size and neg.size else None,
        "wm_count": int(nw), "plain_count": int(np_),
        "success_wm": f"{sw}/{nw}", "success_plain": f"{sp}/{np_}",
        "wm_score_mean": float(pos.mean()) if pos.size else None,
        "plain_score_mean": float(neg.mean()) if neg.size else None,
    }


def _fmt(v):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/vla/eval_out_compression")
    ap.add_argument("--out", default="/workspace/vla/RESULTS_compression.md")
    args = ap.parse_args()
    base = pathlib.Path(args.base)

    rows = []
    # openpi pi0.5: two scopes — whole-model (VLM backbone + action expert) and action-only
    # (action expert + projection heads only; VLM backbone untouched). See
    # build_compressed_ckpt.py --scope. Action-only rows show dashes until that eval runs.
    for suf, tag in (("", "whole-model"), ("_actiononly", "action-only")):
        for atk in ("prune30", "quant"):
            r = openpi_libero("goal", atk, base, suf)
            r["combo"] = f"openpi pi0.5 / LIBERO goal / {atk} ({tag})"; rows.append(r)
        for atk in ("prune30", "quant"):
            r = openpi_robotwin(atk, base, suf)
            r["combo"] = f"openpi pi0.5 / RoboTwin 10-task / {atk} ({tag})"; rows.append(r)
    # lingbot Wan is inherently action-scoped: only the diffusion transformer (the action
    # generator) is compressed; the text encoder / VAE / tokenizer are the untouched base.
    for atk in ("prune", "quant"):
        r = lingbot_libero(atk, base); r["combo"] = f"lingbot Wan / LIBERO-10 / {atk} (transformer-only)"; rows.append(r)
    for atk in ("prune", "quant"):
        r = lingbot_robotwin(atk, base); r["combo"] = f"lingbot Wan / RoboTwin 10-task / {atk} (transformer-only)"; rows.append(r)

    lines = ["# Compression Robustness (§12.5 prune + quant)", ""]
    lines.append("_AUC = per-episode (|G|=1); AUC@16 = group budget |G|=16 (MC mean-aggregate), "
                 "same unit as the §6 main table._")
    lines.append("")
    lines.append("| Combo | AUC | AUC@16 | TPR@1% | TPR@10% | n_wm | n_plain | succ_wm | succ_plain | wm-plain Δscore |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        delta = r.get("wm_minus_plain_mean")
        if delta is None and r.get("wm_score_mean") is not None and r.get("plain_score_mean") is not None:
            delta = r["wm_score_mean"] - r["plain_score_mean"]
        lines.append(
            "| {combo} | {auc} | {auc16} | {tpr1} | {tpr10} | {nw} | {np_} | {sw} | {sp} | {d} |".format(
                combo=r["combo"], auc=_fmt(r.get("auc")), auc16=_fmt(r.get("auc16")),
                tpr1=_fmt(r.get("tpr_1")), tpr10=_fmt(r.get("tpr_10")),
                nw=_fmt(r.get("wm_count")), np_=_fmt(r.get("plain_count")),
                sw=_fmt(r.get("success_wm")), sp=_fmt(r.get("success_plain")), d=_fmt(delta),
            )
        )
    out = pathlib.Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
