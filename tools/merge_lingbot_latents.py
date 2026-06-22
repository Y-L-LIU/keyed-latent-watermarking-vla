"""Merge several LingBot latent datasets into one (e.g. libero_goal + libero_spatial -> libero_goalspatial).

A latent dataset (produced by wan_va/tools/extract_latents.py) has:
  latents/chunk-000/<camera>/episode_<NNNNNN>_<start>_<end>.pth
  data/chunk-000/episode_<NNNNNN>.parquet        (state + action, the training dataloader reads this)
  meta/episodes.jsonl
  empty_emb.pt
This concatenates episodes across inputs with a global episode renumbering, fixing the episode index
in: the latent filenames, the parquet `episode_index` column + filename, and episodes.jsonl. empty_emb
is copied once (instruction-independent).

Run AFTER the GPU extraction of each suite. CPU only.
  python3 merge_lingbot_latents.py --out /workspace/vla/lingbot_latents/libero_goalspatial \
      --inputs /workspace/vla/lingbot_latents/_goal /workspace/vla/lingbot_latents/_spatial
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

import pyarrow.parquet as pq
import pyarrow as pa

EP_RE = re.compile(r"episode_(\d+)")


def _renum_name(name: str, new_idx: int) -> str:
    # episode_000003_0_388.pth -> episode_<new>_0_388.pth ; episode_000003.parquet -> episode_<new>.parquet
    return EP_RE.sub(lambda m: f"episode_{new_idx:06d}", name, count=1)


def merge(out: pathlib.Path, inputs: list[pathlib.Path]) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "latents" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    global_ep = 0
    ep_lines: list[str] = []
    copied_empty = False
    for src in inputs:
        if not copied_empty and (src / "empty_emb.pt").exists():
            shutil.copy2(src / "empty_emb.pt", out / "empty_emb.pt")
            copied_empty = True
        # episodes.jsonl drives the per-source episode order
        src_eps = [json.loads(l) for l in open(src / "meta" / "episodes.jsonl")]
        for rec in src_eps:
            old = int(rec["episode_index"])
            new = global_ep
            # latents: per camera
            for cam_dir in sorted((src / "latents" / "chunk-000").glob("*")):
                if not cam_dir.is_dir():
                    continue
                (out / "latents" / "chunk-000" / cam_dir.name).mkdir(parents=True, exist_ok=True)
                for pth in cam_dir.glob(f"episode_{old:06d}_*.pth"):
                    shutil.copy2(pth, out / "latents" / "chunk-000" / cam_dir.name / _renum_name(pth.name, new))
            # data parquet: renumber episode_index column + filename
            pq_old = src / "data" / "chunk-000" / f"episode_{old:06d}.parquet"
            if pq_old.exists():
                t = pq.read_table(pq_old)
                if "episode_index" in t.column_names:
                    col = pa.array([new] * t.num_rows, type=t.schema.field("episode_index").type)
                    t = t.set_column(t.column_names.index("episode_index"), "episode_index", col)
                pq.write_table(t, out / "data" / "chunk-000" / f"episode_{new:06d}.parquet")
            rec["episode_index"] = new
            ep_lines.append(json.dumps(rec))
            global_ep += 1
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        f.write("\n".join(ep_lines) + "\n")
    # copy a representative info.json/tasks.jsonl from the first input (schema is identical)
    for fn in ("info.json", "tasks.jsonl", "episodes_stats.jsonl"):
        srcf = inputs[0] / "meta" / fn
        if srcf.exists():
            shutil.copy2(srcf, out / "meta" / fn)
    print(f"merged {global_ep} episodes from {len(inputs)} datasets -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--inputs", type=pathlib.Path, nargs="+", required=True)
    a = ap.parse_args()
    merge(a.out, a.inputs)


if __name__ == "__main__":
    main()
