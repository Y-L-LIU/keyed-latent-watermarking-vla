"""Build a re-indexed LingBot latent-dataset skeleton spanning a SPARSE set of original
episodes, renumbered to a CONTIGUOUS 0..N-1 (required because the loader's episode_data_index
is POSITIONAL and _get_global_idx does ["from"][episode_index]). Per-file-symlinks the latents
(episode_{new}_0_{len}.pth -> orig episode_{orig}_0_{len}.pth) and renumbers meta. For the clean
arm it also copies the original action parquets (re-indexed). data/ for watermarked arms is filled
by relabel_pathb.py --reindex.

Usage:
  python3.11 reindex_skeleton.py --out <dir> [--copy-clean]
  (orig episode set = 10 per task x 10 tasks, hardcoded below)
"""
from __future__ import annotations
import argparse, json, os, shutil
import pandas as pd

SRC = "/workspace/vla/lingbot_latents/libero_long"
CAMS = ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
# 10 episodes per task x 10 tasks (each task = a contiguous 50-block); spans all 10 LIBERO tasks.
TASK_STARTS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
NPER = 10
ORIG_EPS = [t + i for t in TASK_STARTS for i in range(NPER)]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-clean", action="store_true", help="also copy original parquets (re-indexed) -> clean arm")
    args = ap.parse_args()
    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data", "chunk-000"))
    os.makedirs(os.path.join(out, "meta"))
    os.symlink(os.path.join(SRC, "empty_emb.pt"), os.path.join(out, "empty_emb.pt"))
    for cam in CAMS:
        os.makedirs(os.path.join(out, "latents", "chunk-000", cam))
        os.makedirs(os.path.join(out, "videos", "chunk-000", cam))

    ep_by_idx = {d["episode_index"]: d for d in read_jsonl(os.path.join(SRC, "meta", "episodes.jsonl"))}
    stat_by_idx = {d["episode_index"]: d for d in read_jsonl(os.path.join(SRC, "meta", "episodes_stats.jsonl"))}
    shutil.copyfile(os.path.join(SRC, "meta", "tasks.jsonl"), os.path.join(out, "meta", "tasks.jsonl"))

    new_eps, new_stats, total_frames = [], [], 0
    for new_i, orig in enumerate(ORIG_EPS):
        ep = json.loads(json.dumps(ep_by_idx[orig])); ep["episode_index"] = new_i
        new_eps.append(ep)
        length = int(ep["length"]); total_frames += length
        st = json.loads(json.dumps(stat_by_idx[orig])); st["episode_index"] = new_i
        new_stats.append(st)
        ac = ep["action_config"][0]; s0, e0 = int(ac["start_frame"]), int(ac["end_frame"])
        for cam in CAMS:
            src_lat = os.path.join(SRC, "latents", "chunk-000", cam, f"episode_{orig:06d}_{s0}_{e0}.pth")
            dst_lat = os.path.join(out, "latents", "chunk-000", cam, f"episode_{new_i:06d}_{s0}_{e0}.pth")
            if not os.path.exists(src_lat):
                raise FileNotFoundError(src_lat)
            os.symlink(src_lat, dst_lat)
            # symlink video too (harmless; relabel reads ORIG videos directly)
            sv = os.path.join(SRC, "videos", "chunk-000", cam, f"episode_{orig:06d}.mp4")
            if os.path.exists(sv):
                os.symlink(sv, os.path.join(out, "videos", "chunk-000", cam, f"episode_{new_i:06d}.mp4"))
        if args.copy_clean:
            df = pd.read_parquet(os.path.join(SRC, "data", "chunk-000", f"episode_{orig:06d}.parquet"))
            df = df.reset_index(drop=True)
            df["episode_index"] = new_i
            df["index"] = range(total_frames - length, total_frames)  # contiguous global index
            df.to_parquet(os.path.join(out, "data", "chunk-000", f"episode_{new_i:06d}.parquet"))

    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        for e in new_eps:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as f:
        for s in new_stats:
            f.write(json.dumps(s) + "\n")
    info = json.load(open(os.path.join(SRC, "meta", "info.json")))
    info["total_episodes"] = len(ORIG_EPS)
    info["total_frames"] = total_frames
    info["total_videos"] = len(ORIG_EPS) * len(CAMS)
    info["splits"] = {"train": f"0:{len(ORIG_EPS)}"}
    json.dump(info, open(os.path.join(out, "meta", "info.json"), "w"), indent=4)
    print(f"[reindex] {out}: {len(ORIG_EPS)} eps (10/task x10), total_frames={total_frames}, "
          f"clean_parquets={'yes' if args.copy_clean else 'no'}")


if __name__ == "__main__":
    main()
