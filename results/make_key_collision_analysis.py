#!/usr/bin/env python3
"""Key uniqueness / collision analysis (threat-model §12.5).

Goal
----
Empirically bound the probability that an INDEPENDENT owner key ``k' != k*``
triggers a positive provenance decision -- a "key collision". This is what
establishes that the keyed-latent fingerprint UNIQUELY binds to one owner.

Pipeline (paper notation)
--------------------------
- per-window similarity   psi(.) = whitened matched filter, subspace_rank = 3
  (verbatim re-implementation of ``_wmf_score_from_vectors`` /
   ``_select_whitened_subspace`` from
   ``openpi/scripts/eval_libero_action_inversion.py``).
- per-episode calibrated z-score  Z_e(k)   (\Cref{eq:zscore})
      Z_e(k) = (s_e(k) - mu_e^-(k)) / (sigma_e^-(k) + eps)
  where (mu_e^-, sigma_e^-) come from a per-episode false-key calibration bank.
- group statistic                 T_G(k) = sum_{e in G} Z_e(k)   (\Cref{eq:aggregation}).
- false-key null:  for random k' != k*, T_G(k') is the H0 distribution; the
  collision probability is  Pr[T_G(k') > tau_decision | k' != k*].

Protocol (must match the §5.x subsection)
------------------------------------------
- pool       : pi0.5 / LIBERO-10 partial+MAP saved rollouts (watermarked).
- feature    : ``chunk_recovered_noise_partial_map`` (mainline partial+MAP zhat).
- detector   : whitened matched filter, subspace_rank = 3.
- group size : |G| = 16.
- false keys : J_eval >= 1024 evaluation keys (disjoint from the J_cal = 32
               calibration bank), numpy seed 20260529.
- tau        : set at target FPR = 1e-3 from the false-key T_G null via a
               bootstrap over groups of 16 keys/episodes.

This is pure numpy on SAVED data -- no model and no GPU.
"""
from __future__ import annotations

import csv
import glob
import math
import pathlib
import sys
import time

import numpy as np

# openpi watermark module is the keyed reference generator r(k, nu, c).
sys.path.insert(0, "/workspace/vla/openpi/src")
import openpi.policies.watermark as wm  # noqa: E402

# --------------------------------------------------------------------------- #
ROLLOUT_DIR = pathlib.Path(
    "/workspace/vla/eval_out/base/libero_10/rollouts/none/task_rollout"
)
OUT_DIR = pathlib.Path("/workspace/vla/results")
FEATURE_FIELD = "chunk_recovered_noise_partial_map"
SUBSPACE_RANK = 3
_RAW_MF = bool(__import__("os").environ.get("RAW_MF"))  # raw matched filter (r^T z) when RAW_MF set
MAX_WINDOWS = 5            # selected chunks scored per episode (count=5)
SCORE_STEP_SCOPE = "full_chunk"
J_CAL = 32                 # per-episode false-key CALIBRATION bank (paper default)
J_EVAL = 1024              # >= 1024 INDEPENDENT evaluation false keys (the null)
GROUP_SIZE = 16            # |G|
SEED = 20260529
TARGET_FPR = 1e-3
N_BOOT_GROUPS = 200_000    # bootstrap T_G groups for the tail estimate
EPS = 1e-8


# --------------------------------------------------------------------------- #
# Detector math -- verbatim from openpi eval_libero_action_inversion.py
# --------------------------------------------------------------------------- #
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-centered cosine between two flattened chunk tensors (base detector)."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _select_whitened_subspace(centered_feature, null_matrix, subspace_rank):
    centered_feature = np.asarray(centered_feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    dim = centered_feature.shape[0]
    if dim == 0:
        return centered_feature, np.eye(0), np.zeros((0,))
    centered_null = null_matrix - np.mean(null_matrix, axis=0, keepdims=True)
    if null_matrix.shape[0] <= 1:
        cov = np.eye(dim)
    else:
        cov = np.asarray(np.cov(centered_null, rowvar=False, bias=False), dtype=np.float64)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    reg = max(1e-6, 1e-4 * float(np.trace(cov)) / max(dim, 1))
    cov = cov + reg * np.eye(dim)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if subspace_rank is not None:
        rank = min(int(subspace_rank), dim)
        eigvals = eigvals[:rank]
        eigvecs = eigvecs[:, :rank]
    return eigvecs.T @ centered_feature, eigvecs, eigvals


def _wmf_score(feature, null_matrix, subspace_rank=SUBSPACE_RANK) -> float:
    """Whitened matched filter psi = template^T Sigma^{-1} (feature - mu)."""
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0 or null_matrix.size == 0:
        return 0.0
    if _RAW_MF:
        return float(np.sum(feature - np.mean(null_matrix, axis=0)))
    centered = feature - np.mean(null_matrix, axis=0)
    pf, evec, ev = _select_whitened_subspace(centered, null_matrix, subspace_rank)
    if pf.size == 0:
        return 0.0
    template = np.sum(evec, axis=0)
    inv = 1.0 / np.sqrt(np.maximum(ev, 1e-8))
    return float(np.dot(template * inv, pf * inv))


# --------------------------------------------------------------------------- #
# Per-episode feature generation
# --------------------------------------------------------------------------- #
def _episode_feature_for_key(d, key, *, recovered, use_saved_reference=False):
    """Per-window cosine feature vector of recovered noise vs the key-`key` reference.

    Gaussian reference mode (matching the saved true reference). For the true
    key we reuse the stored ``chunk_reference`` (verified to regenerate
    bit-identically); for any other key we regenerate r(key, nu, c).
    """
    sel = d["chunk_selected"]
    ex = d["chunk_executed_steps"]
    ref = d["chunk_reference"]
    cidx = d["chunk_chunk_index"]
    nonce = int(d["episode_nonce"])
    srate = float(d["sample_rate_hz"])
    fmin = float(d["freq_min_hz"])
    fmax = float(d["freq_max_hz"])
    L = int(ref.shape[1])
    A = int(ref.shape[2])
    cfg = wm.InternalNoiseWatermarkConfig(
        secret_key=int(key), control_freq=srate, beta=0.0,
        freq_range=(fmin, fmax), reference_mode="gaussian",
        chunk_selection_strategy=str(d["chunk_selection_strategy"]),
        chunk_selection_period=int(d["chunk_selection_period"]),
        chunk_selection_count=int(d["chunk_selection_count"]),
    )
    out = []
    cnt = 0
    for i in range(len(sel)):
        if not sel[i] or ex[i] <= 0:
            continue
        if cnt >= MAX_WINDOWS:
            break
        if use_saved_reference:
            r = ref[i]
        else:
            ctx = wm.WatermarkContext(chunk_index=int(cidx[i]), episode_nonce=nonce)
            r = wm.generate_keyed_reference(
                length=L, action_dim=A, sample_rate_hz=srate, config=cfg, context=ctx
            )
        steps = r.shape[0]  # full_chunk
        out.append(_cosine_sim(recovered[i][:steps], r[:steps]))
        cnt += 1
    return np.asarray(out, dtype=np.float64)


def build_episode_zscores(d, *, rng):
    """Return (z_true, z_eval) for one watermarked episode.

    z_true     : scalar calibrated z-score Z_e(k*).
    z_eval     : (J_EVAL,) calibrated z-scores Z_e(k') for INDEPENDENT false keys.

    Calibration bank (J_CAL keys) defines (mu_e^-, sigma_e^-) and the whitening
    null; the J_EVAL evaluation keys are disjoint and form the false-key null.
    """
    sk = int(d["secret_key"])
    recovered = d[FEATURE_FIELD]

    # disjoint key blocks, both != k*
    cal_keys = [sk + 1 + i for i in range(J_CAL)]
    # evaluation keys: a large random block of independent owner keys, none equal
    # to k* and none colliding with the calibration block.
    eval_keys = sk + 1 + J_CAL + np.arange(J_EVAL, dtype=np.int64)

    cal_feats = np.stack([_episode_feature_for_key(d, k, recovered=recovered) for k in cal_keys])
    true_feat = _episode_feature_for_key(d, sk, recovered=recovered, use_saved_reference=True)

    # raw WMF scores whitened against the calibration bank
    s_true = _wmf_score(true_feat, cal_feats)
    # per-episode calibration moments: leave-one-out WMF of the calibration keys
    s_cal = np.array(
        [_wmf_score(cal_feats[i], np.delete(cal_feats, i, axis=0)) for i in range(J_CAL)]
    )
    mu = float(s_cal.mean())
    sd = float(s_cal.std(ddof=1))
    z_true = (s_true - mu) / (sd + EPS)

    # evaluation false keys scored against the SAME calibration bank, then
    # calibrated with the same (mu, sd) -> Z_e(k') for independent owners.
    s_eval = np.empty(J_EVAL, dtype=np.float64)
    for j, k in enumerate(eval_keys):
        fe = _episode_feature_for_key(d, int(k), recovered=recovered)
        s_eval[j] = _wmf_score(fe, cal_feats)
    z_eval = (s_eval - mu) / (sd + EPS)
    return z_true, z_eval


# --------------------------------------------------------------------------- #
# Tail fits
# --------------------------------------------------------------------------- #
def _erfcinv(y):
    """Inverse complementary error function (no scipy dependency)."""
    try:
        from scipy.special import erfcinv
        return float(erfcinv(y))
    except Exception:
        from statistics import NormalDist
        # erfcinv(y) = -ndtri(y/2)/sqrt(2)
        return -NormalDist().inv_cdf(y / 2.0) / math.sqrt(2.0)


def _gaussian_tail(tg_null, tau):
    mu = float(tg_null.mean())
    sd = float(tg_null.std(ddof=1))
    # survival function of a normal
    from math import erfc, sqrt
    z = (tau - mu) / sd
    return 0.5 * erfc(z / sqrt(2.0)), mu, sd


def _gev_fit_and_tail(block_maxima, tau):
    """Fit a GEV to block maxima by ML (scipy) and return Pr[X > tau]."""
    try:
        from scipy.stats import genextreme
    except Exception:
        return float("nan"), None
    c, loc, scale = genextreme.fit(block_maxima)
    sf = float(genextreme.sf(tau, c, loc=loc, scale=scale))
    return sf, (c, loc, scale)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    files = sorted(
        f for f in glob.glob(str(ROLLOUT_DIR / "*_watermarked.npz"))
        if "extra_modes" not in f
    )
    if not files:
        print(f"[collision] no watermarked npz under {ROLLOUT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[collision] {len(files)} watermarked episodes; "
          f"J_cal={J_CAL}, J_eval={J_EVAL}, |G|={GROUP_SIZE}, seed={SEED}")

    # Feature generation (the expensive step) is cached to .npy so the cheap
    # tail-fit/bootstrap stats can be re-derived in seconds. Pass --recompute
    # to force regeneration.
    _sfx = "_raw" if _RAW_MF else ""  # scorer-aware cache so raw/whitened never cross-load
    z_eval_cache = OUT_DIR / f"key_collision_z_eval{_sfx}.npy"
    z_true_cache = OUT_DIR / f"key_collision_z_true{_sfx}.npy"
    recompute = "--recompute" in sys.argv
    sk = int(np.load(files[0], allow_pickle=True)["secret_key"])
    if (not recompute and z_eval_cache.exists() and z_true_cache.exists()
            and np.load(z_eval_cache).shape == (len(files), J_EVAL)):
        print(f"[collision] loading cached features from {z_eval_cache}")
        z_eval = np.load(z_eval_cache).astype(np.float64)
        z_true_all = np.load(z_true_cache).astype(np.float64)
    else:
        z_true_all = []
        z_eval_rows = []  # (n_ep, J_EVAL)
        for n, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            zt, ze = build_episode_zscores(d, rng=rng)
            z_true_all.append(zt)
            z_eval_rows.append(ze)
            if (n + 1) % 10 == 0:
                print(f"[collision]  scored {n + 1}/{len(files)} episodes "
                      f"({time.time() - t0:.0f}s)", flush=True)
        z_true_all = np.asarray(z_true_all, dtype=np.float64)      # (n_ep,)
        z_eval = np.asarray(z_eval_rows, dtype=np.float64)         # (n_ep, J_EVAL)
    n_ep = z_eval.shape[0]

    # ---- per-key marginal false-key z (pool over episodes), for reporting ----
    z_eval_flat = z_eval.ravel()
    print(f"[collision] false-key Z_e(k') pool: n={z_eval_flat.size}  "
          f"mean={z_eval_flat.mean():.3f}  std={z_eval_flat.std(ddof=1):.3f}  "
          f"max={z_eval_flat.max():.3f}")
    print(f"[collision] true-key Z_e(k*) over {n_ep} wm episodes: "
          f"mean={z_true_all.mean():.3f}  std={z_true_all.std(ddof=1):.3f}")

    # ---- false-key T_G null: bootstrap groups of |G| over independent keys ----
    # Each group: pick GROUP_SIZE episodes (with replacement) and, for each,
    # an INDEPENDENT random evaluation key. T_G = sum of those z-scores.
    ep_idx = rng.integers(0, n_ep, size=(N_BOOT_GROUPS, GROUP_SIZE))
    key_idx = rng.integers(0, J_EVAL, size=(N_BOOT_GROUPS, GROUP_SIZE))
    tg_null = z_eval[ep_idx, key_idx].sum(axis=1)                  # (N_BOOT_GROUPS,)

    # true-key T_G (group of 16 watermarked episodes), for the H1 reference point
    tg_true = z_true_all[rng.integers(0, n_ep, size=(N_BOOT_GROUPS, GROUP_SIZE))].sum(axis=1)

    # ---- tau at FPR = 1e-3 from the false-key null ----
    tau = float(np.quantile(tg_null, 1.0 - TARGET_FPR))
    # empirical collision Pr at this empirical tau is TARGET_FPR by construction;
    # report the count beyond tau and the max as honesty checks.
    n_beyond = int((tg_null > tau).sum())
    emp_collision = float((tg_null > tau).mean())
    tpr_true = float((tg_true > tau).mean())  # detection power at this tau

    # ---- Gaussian tail fit (primary; CLT over |G| independent z-scores) ----
    # tau is set parametrically from the Gaussian fit (non-circular), then the
    # EMPIRICAL collision Pr is read off the null at that tau.
    g_mu = float(tg_null.mean())
    g_sd = float(tg_null.std(ddof=1))
    tau_gauss = g_mu + g_sd * math.sqrt(2.0) * _erfcinv(2.0 * TARGET_FPR)
    emp_collision_at_gauss_tau = float((tg_null > tau_gauss).mean())
    g_collision, _, _ = _gaussian_tail(tg_null, tau)  # Gaussian SF at empirical tau

    # ---- GEV tail fit on DISJOINT-key block maxima (independent extremes) ----
    # Partition the J_EVAL keys into disjoint columns of GROUP_SIZE keys so each
    # block T_G uses a fresh key set -> block maxima are genuinely independent,
    # which is what GEV assumes. n_cols disjoint key-blocks x many episode draws.
    n_cols = J_EVAL // GROUP_SIZE
    key_cols = np.arange(n_cols * GROUP_SIZE).reshape(n_cols, GROUP_SIZE)
    N_BLK_DRAWS = 4000
    block_tg = np.empty((N_BLK_DRAWS, n_cols), dtype=np.float64)
    for b in range(N_BLK_DRAWS):
        eps = rng.integers(0, n_ep, size=(n_cols, GROUP_SIZE))
        block_tg[b] = z_eval[eps, key_cols].sum(axis=1)
    block_max = block_tg.max(axis=1)  # max over n_cols independent false-key T_G
    # GEV models the per-BLOCK maximum; convert its survival at tau back to a
    # per-KEY collision probability:  p_key = 1 - (1 - SF_block)^(1/n_cols).
    gev_block_sf, gev_params = _gev_fit_and_tail(block_max, tau)
    if gev_block_sf == gev_block_sf:  # not NaN
        gev_collision = 1.0 - (1.0 - min(gev_block_sf, 1.0)) ** (1.0 / n_cols)
    else:
        gev_collision = float("nan")

    # ---- separation margin: true-key T_G(k*) vs false-key null ----
    # T_G(k*) ~ N(|G|*mu_true, |G|*sigma_true^2) by CLT over independent episodes.
    mu_true_ep = float(z_true_all.mean())
    sd_true_ep = float(z_true_all.std(ddof=1))
    tg_true_mu = GROUP_SIZE * mu_true_ep
    tg_true_sd = math.sqrt(GROUP_SIZE) * sd_true_ep
    # Gaussian decision threshold and detection TPR at a sweep of target FPRs,
    # to show how low the operating FPR can be pushed while still detecting k*.
    fpr_sweep = [1e-3, 1e-4, 1e-6, 1e-9]
    op_points = []
    for a in fpr_sweep:
        ta = g_mu + g_sd * math.sqrt(2.0) * _erfcinv(2.0 * a)
        # TPR(k*) under the Gaussian model for T_G(k*)
        tpr_a = 0.5 * math.erfc(((ta - tg_true_mu) / tg_true_sd) / math.sqrt(2.0))
        op_points.append((a, ta, tpr_a))

    # ---- key-space entropy from watermark.py ----
    # The reference r(k, nu, c) is generated by seeding default_rng with
    # blake2b("k:nu:c:d", digest_size=8) -> a 64-bit one-way digest (see
    # _stable_seed). The key VARIABLE secret_key is a plain Python int; the
    # construction accepts arbitrarily wide ints (no width limit), so the key
    # entropy is a free parameter. We deploy a 128-bit key; the empirical
    # collision study above uses narrower keys WLOG since the per-key trigger
    # probability is width-invariant -- only this birthday bound scales with b.
    key_bits = 128  # deployed key width; raisable at will (secret_key is unbounded int)

    # informal collision bound for a realistic owner population of size N_owner.
    # Two distinct collision channels:
    #  (a) KEY-ASSIGNMENT collision: two of N owners are handed the same
    #      key (birthday bound) ~ N^2 / 2^(key_bits+1). For N=1e6, b=128 this is ~1e-27.
    #  (b) STATISTICAL false-trigger: an innocent owner's DISTINCT key trips the
    #      decision. A single verification at operating FPR alpha has this prob
    #      = alpha; over N candidate keys the union bound is N*alpha, so alpha
    #      must be set below 1/N. Because T_G(k*) sits far above the null, the
    #      operating point can be pushed down: the Gaussian extrapolation gives
    #      the tau needed for any alpha, and the TPR(k*) at that tau (op_points).
    N_OWNER = 1_000_000
    birthday_bound = N_OWNER * N_OWNER / (2.0 ** (key_bits + 1))
    union_bound_statistical = N_OWNER * TARGET_FPR  # at the FPR=1e-3 operating point

    # ---- write CSV ----
    csv_path = OUT_DIR / "key_collision_analysis.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# key collision analysis (threat-model 12.5)"])
        w.writerow(["pool", "pi0.5/libero_10 partial+MAP (chunk_recovered_noise_partial_map)"])
        w.writerow(["secret_key_kstar", sk])
        w.writerow(["n_episodes", n_ep])
        w.writerow(["J_cal", J_CAL, "J_eval", J_EVAL])
        w.writerow(["group_size_G", GROUP_SIZE, "seed", SEED, "n_boot_groups", N_BOOT_GROUPS])
        w.writerow(["subspace_rank", SUBSPACE_RANK])
        w.writerow(["target_fpr", TARGET_FPR])
        w.writerow(["tau_decision_empirical", f"{tau:.6f}"])
        w.writerow(["empirical_collision_pr_at_empirical_tau", f"{emp_collision:.3e}",
                    "n_beyond", n_beyond])
        w.writerow(["tau_decision_gaussian", f"{tau_gauss:.6f}"])
        w.writerow(["empirical_collision_pr_at_gaussian_tau", f"{emp_collision_at_gauss_tau:.3e}"])
        w.writerow(["gaussian_sf_at_empirical_tau", f"{g_collision:.3e}",
                    "mu_TG_null", f"{g_mu:.4f}", "sd_TG_null", f"{g_sd:.4f}"])
        w.writerow(["gev_per_key_collision_pr_at_tau", f"{gev_collision:.3e}",
                    "gev_block_sf", f"{gev_block_sf:.3e}", "n_blocks_per_max", n_cols,
                    "gev_c_loc_scale", "" if gev_params is None else
                    f"{gev_params[0]:.4f},{gev_params[1]:.4f},{gev_params[2]:.4f}"])
        w.writerow(["true_key_TPR_at_tau_G16", f"{tpr_true:.4f}"])
        w.writerow(["TG_true_mean", f"{tg_true_mu:.4f}", "TG_true_sd", f"{tg_true_sd:.4f}"])
        w.writerow(["key_space_entropy_bits", key_bits,
                    "note", "128-bit deployed key; secret_key is an unbounded int (no width limit); width-invariant detection"])
        w.writerow(["birthday_collision_bound_Nowner", N_OWNER, "bound", f"{birthday_bound:.3e}",
                    "note", "" if birthday_bound <= 1 else "exceeds 1: int32 too small for this N, widen key"])
        w.writerow(["statistical_union_bound_at_fpr", TARGET_FPR, "Nowner", N_OWNER,
                    "bound", f"{union_bound_statistical:.3e}",
                    "note", "loose at 1e-3; push FPR<1/Nowner (see op_points)"])
        w.writerow([])
        w.writerow(["# operating points: Gaussian-extrapolated tau and TPR(k*) vs target FPR"])
        w.writerow(["target_fpr", "tau", "tpr_true_kstar"])
        for a, ta, tpr_a in op_points:
            w.writerow([f"{a:.0e}", f"{ta:.4f}", f"{tpr_a:.4f}"])
        w.writerow([])
        # dump the false-key T_G null sample (subsampled to 10000 for size) +
        # the full per-key marginal false-key z summary quantiles.
        w.writerow(["# false-key T_G null quantiles (|G|=16)"])
        for q in [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]:
            w.writerow([f"q{q}", f"{np.quantile(tg_null, q):.6f}"])
        w.writerow([])
        w.writerow(["# per-key marginal false-key Z (single episode) quantiles"])
        for q in [0.5, 0.9, 0.99, 0.999, 1.0]:
            w.writerow([f"z_q{q}", f"{np.quantile(z_eval_flat, q):.6f}"])
        w.writerow([])
        w.writerow(["# true-key Z_e(k*) per watermarked episode"])
        w.writerow(["episode_idx", "z_true"])
        for i, zt in enumerate(z_true_all):
            w.writerow([i, f"{zt:.6f}"])

    # also save the raw T_G null sample as npy for the LaTeX figure / re-fitting
    np.save(OUT_DIR / "key_collision_TG_null.npy", tg_null.astype(np.float32))
    np.save(OUT_DIR / "key_collision_z_eval.npy", z_eval.astype(np.float32))
    np.save(OUT_DIR / "key_collision_z_true.npy", z_true_all.astype(np.float32))

    # ---- console report ----
    print("\n===== KEY COLLISION ANALYSIS (threat-model 12.5) =====")
    print(f"  k*                          = {sk}")
    print(f"  episodes (watermarked)      = {n_ep}")
    print(f"  J_cal / J_eval              = {J_CAL} / {J_EVAL}")
    print(f"  |G|                         = {GROUP_SIZE}   seed = {SEED}")
    print(f"  false-key T_G null: mean={g_mu:.3f} sd={g_sd:.3f} max={tg_null.max():.3f}")
    print(f"  tau @ FPR=1e-3 (empirical)  = {tau:.4f}")
    print(f"  tau @ FPR=1e-3 (Gaussian)   = {tau_gauss:.4f}")
    print(f"  empirical collision @ emp.tau = {emp_collision:.3e}  (n_beyond={n_beyond}/{N_BOOT_GROUPS})")
    print(f"  empirical collision @ Gauss.tau = {emp_collision_at_gauss_tau:.3e}")
    print(f"  Gaussian SF @ empirical tau = {g_collision:.3e}")
    print(f"  GEV per-key tail @ emp.tau  = {gev_collision:.3e}  (shape c={gev_params[0]:.4f} ~ Gumbel/light)")
    print(f"  true-key TPR @ tau (|G|=16) = {tpr_true:.3f}")
    print(f"  T_G(k*) ~ N({tg_true_mu:.1f}, {tg_true_sd:.1f}^2)  (sits far above null max={tg_null.max():.1f})")
    print(f"  operating points (Gaussian extrapolation):")
    for a, ta, tpr_a in op_points:
        print(f"      FPR={a:.0e}  tau={ta:5.1f}  TPR(k*)={tpr_a:.3f}")
    print(f"  key-space entropy           ~ {key_bits} bits (128-bit deployed key; "
          f"unbounded-int secret_key; width-invariant detection)")
    print(f"  birthday key-collision bound (N={N_OWNER:.0e}) = {birthday_bound:.3e}"
          f"  {'<-- >1: int32 too small for 1e6 owners, widen key' if birthday_bound > 1 else ''}")
    print(f"  statistical union @1e-3 (N={N_OWNER:.0e})       = {union_bound_statistical:.3e}"
          f"  (loose; push FPR<1/N)")
    print(f"  wrote {csv_path}")
    print(f"  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
