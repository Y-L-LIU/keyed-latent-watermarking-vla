"""Per-window rescore of saved robotwin watermark MAP .npz files."""

import argparse
import glob
import os

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def wmf_score(noise: np.ndarray, reference: np.ndarray) -> float:
    x = noise.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    n = min(x.size, r.size)
    if n == 0:
        return 0.0
    x, r = x[:n], r[:n]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return 0.0
    proj = np.dot(x, r) / r_norm
    return float(proj * np.sqrt(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_dir", help="Directory containing episode_*.npz files")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "episode_*.npz")))
    if not files:
        print(f"No .npz files found in {args.npz_dir}")
        return

    for path in files:
        data = np.load(path, allow_pickle=True)
        ep = int(data["episode_idx"])
        variant = str(data["variant"])
        selected = data["chunk_selected"]
        executed_steps = data["chunk_executed_steps"]
        recovered = data["chunk_recovered_noise"]
        reference = data["chunk_reference"]
        n_windows = len(selected)

        print(f"\n{'='*80}")
        print(f"Episode {ep} | {variant} | {n_windows} windows | selected: {selected.sum()}")
        print(f"{'='*80}")
        print(f"{'win':>4} {'sel':>4} {'exec':>5} "
              f"{'cos_exec':>10} {'cos_full':>10} "
              f"{'wmf_exec':>10} {'wmf_full':>10}")
        print("-" * 70)

        cos_exec_sel, cos_full_sel = [], []
        wmf_exec_sel, wmf_full_sel = [], []

        for w in range(n_windows):
            sel = bool(selected[w])
            es = int(executed_steps[w])
            rec = recovered[w]   # (50, 32)
            ref = reference[w]   # (50, 32)

            cos_e = cosine_sim(rec[:es], ref[:es])
            cos_f = cosine_sim(rec, ref)
            wmf_e = wmf_score(rec[:es], ref[:es])
            wmf_f = wmf_score(rec, ref)

            marker = " *" if sel else ""
            print(f"{w:>4} {'Y' if sel else 'N':>4} {es:>5} "
                  f"{cos_e:>10.4f} {cos_f:>10.4f} "
                  f"{wmf_e:>10.2f} {wmf_f:>10.2f}{marker}")

            if sel and es > 0:
                cos_exec_sel.append(cos_e)
                cos_full_sel.append(cos_f)
                wmf_exec_sel.append(wmf_e)
                wmf_full_sel.append(wmf_f)

        if cos_exec_sel:
            print(f"\nSelected windows mean (n={len(cos_exec_sel)}):")
            print(f"  cos_exec={np.mean(cos_exec_sel):.4f}  cos_full={np.mean(cos_full_sel):.4f}")
            print(f"  wmf_exec={np.mean(wmf_exec_sel):.2f}  wmf_full={np.mean(wmf_full_sel):.2f}")


if __name__ == "__main__":
    main()
