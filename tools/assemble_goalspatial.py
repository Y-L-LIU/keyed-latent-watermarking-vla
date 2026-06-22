#!/usr/bin/env python3
"""Assemble the LingBot goal+spatial latent TRAINING dataset as the trainer expects it.

The trainer (MultiLatentLeRobotDataset -> construct_lerobot_multi_processor) does
`recursive_find_file(config.dataset_path, 'info.json')` and treats EVERY meta/info.json
found under dataset_path as an independent LatentLeRobotDataset, concatenating them.
So goal+spatial is NOT a flat global merge: it is two self-contained per-suite datasets
living side by side as subdirs:

    <out>/goal/      (a complete lerobot+latents dataset, episodes 0..499, tasks 0..9)
    <out>/spatial/   (a complete lerobot+latents dataset, episodes 0..499, tasks 0..9)

Each per-suite dataset = the original lerobot source (data/ + meta/info.json + tasks.jsonl
+ episodes_stats.jsonl, with inline images in the parquet) PLUS:
  - meta/episodes.jsonl  taken from the EXTRACTOR output (it carries the `action_config`
    that LatentLeRobotDataset.parse_meta needs and that matches the latent filenames),
  - latents/             from the extractor,
  - empty_emb.pt         from the extractor.
data/ and latents/ are symlinked (large, ~5MB/parquet inline images); the small meta files
and empty_emb are copied so the dataset survives cleanup of the inputs.

NO renumbering is needed (each suite stays independent), which is why this is simpler and
safer than a flat merge.  Run AFTER both suites' extraction has fully completed.

Usage:
  python3 assemble_goalspatial.py --out /workspace/vla/lingbot_latents/libero_goalspatial \
    --suite goal    /workspace/vla/lingbot_latents/_goal    /workspace/vla/lingbot_latents/_src/libero_goal \
    --suite spatial /workspace/vla/lingbot_latents/_spatial /workspace/vla/lingbot_latents/_src/libero_spatial
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil


def _symlink(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src.resolve(), dst)


def _count_lines(p: pathlib.Path) -> int:
    return sum(1 for _ in open(p))


def assemble_suite(out_suite: pathlib.Path, lat: pathlib.Path, src: pathlib.Path) -> int:
    (out_suite / "meta").mkdir(parents=True, exist_ok=True)
    # meta: info/tasks/episodes_stats from the lerobot source; episodes.jsonl from extractor
    for fn in ("info.json", "tasks.jsonl", "episodes_stats.jsonl"):
        shutil.copy2(src / "meta" / fn, out_suite / "meta" / fn)
    shutil.copy2(lat / "meta" / "episodes.jsonl", out_suite / "meta" / "episodes.jsonl")
    shutil.copy2(lat / "empty_emb.pt", out_suite / "empty_emb.pt")
    # large dirs: symlink
    _symlink(src / "data", out_suite / "data")
    _symlink(lat / "latents", out_suite / "latents")

    # consistency checks
    n_ep = _count_lines(out_suite / "meta" / "episodes.jsonl")
    n_pq = len(list((src / "data" / "chunk-000").glob("episode_*.parquet")))
    cams = [d for d in (lat / "latents" / "chunk-000").glob("*") if d.is_dir()]
    n_lat = [len(list(c.glob("episode_*.pth"))) for c in cams]
    info = json.load(open(out_suite / "meta" / "info.json"))
    print(f"  [{out_suite.name}] episodes.jsonl={n_ep} parquet={n_pq} "
          f"latents/cam={dict(zip([c.name for c in cams], n_lat))} info.total_episodes={info['total_episodes']}")
    assert n_ep == n_pq == info["total_episodes"], (
        f"count mismatch: episodes.jsonl={n_ep} parquet={n_pq} info={info['total_episodes']}")
    assert cams and all(n == n_ep for n in n_lat), (
        f"latent count mismatch vs episodes ({n_ep}): {dict(zip([c.name for c in cams], n_lat))}")
    return n_ep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite", action="append", nargs=3, metavar=("NAME", "LATENT_ROOT", "SRC_ROOT"),
                    required=True, help="repeatable: suite name, extracted-latent root, lerobot source root")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, lat, src in a.suite:
        total += assemble_suite(a.out / name, pathlib.Path(lat), pathlib.Path(src))
    # config.empty_emb_path is the TOP-LEVEL dataset_path/empty_emb.pt (instruction-independent),
    # so place a copy there too (the trainer reads this one, not the per-suite copies).
    first_lat = pathlib.Path(a.suite[0][1])
    shutil.copy2(first_lat / "empty_emb.pt", a.out / "empty_emb.pt")
    print(f"assembled {len(a.suite)} per-suite datasets ({total} episodes total) under {a.out}")


if __name__ == "__main__":
    main()
