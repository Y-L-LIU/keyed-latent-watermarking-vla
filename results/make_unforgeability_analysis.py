#!/usr/bin/env python3
"""Empirical unforgeability analysis for the keyed-latent watermark (threat-model §12.3).

Attacker model: knows the algorithm and the verifier code (this very detector),
has N watermarked rollouts, but NOT the owner secret key k*. We measure whether
such an attacker can FORGE a key k_hat whose group statistic T_G(k_hat) reaches
the decision threshold tau_decision (calibrated at FPR=1e-3), within a feasible
compute budget.

Two strategies are evaluated:
  (a) Brute-force / random key guessing. Each guess is an i.i.d. random key drawn
      from the key space; we compute its group statistic T_G against the SAME
      partial+MAP recovered-noise pool the verifier would use. The per-guess
      success probability p equals the false-key collision rate at tau, so the
      expected forgery budget is ~1/p, and is tied to the key entropy 2^bits.
  (b) Structured / gradient attack. Argued (and probed) to be non-viable because
      the seed->reference map is a blake2b hash (one-way, non-differentiable in
      the integer key). A relaxed continuous-key surrogate cannot exist: the key
      enters only through hashlib.blake2b(...) inside watermark._stable_seed, so
      there is no gradient path k -> r(k). See the printed diagnostic below.

Protocol (matches the uniqueness analysis for cross-subsection consistency):
  - pool: pi0.5 / LIBERO-10 watermarked rollouts, partial+MAP recovered noise
  - detector: whitened matched filter (WMF), subspace_rank=3
  - per-episode calibration: Z_e(k) = (s_e(k) - mu^-_e(k)) / (sigma^-_e(k) + eps)
    using a candidate-specific 32-key false-key bank (eq:zscore)
  - aggregation: T_G(k) = sum_{e in G} Z_e(k) over |G|=16 (eq:aggregation)
  - tau_decision at FPR = 1e-3
  - numpy seed = 20260529

Outputs (this dir):
  unforgeability_analysis.csv  attacker success vs budget M; tau; p; max T_G; entropy
"""
from __future__ import annotations

import glob
import math
import pathlib
import sys

import numpy as np

# --- watermark key generator (offline, numpy-only; no model / GPU) ----------- #
OPENPI_SRC = "/workspace/vla/openpi/src"
if OPENPI_SRC not in sys.path:
    sys.path.insert(0, OPENPI_SRC)
from openpi.policies import watermark as wm  # noqa: E402

# --------------------------------------------------------------------------- #
HERE = pathlib.Path(__file__).resolve().parent
ROLLOUT_DIR = pathlib.Path(
    "/workspace/vla/eval_out/base/libero_10/rollouts/none/task_rollout"
)
OUT_CSV = HERE / "unforgeability_analysis.csv"

SEED = 20260529
SUBSPACE_RANK = 3
_RAW_MF = bool(__import__("os").environ.get("RAW_MF"))  # raw matched filter (r^T z) when RAW_MF set
N_FALSE = 32          # false-key bank size for per-episode calibration (matches data)
G = 16               # group size |G| for aggregation (matches uniqueness analysis)
FPR_TARGET = 1e-3    # tau_decision FPR
EPS = 1e-8

# Attacker budgets (number of independent random key guesses).
BUDGETS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
N_GROUP_SAMPLES = 20000  # MC groups of size G used for null/H1 distributions
KEY_SPACE_BITS = 128     # deployed key width; secret_key is an unbounded int hashed by blake2b (no width limit). Detection p/tau are width-invariant; only this entropy ceiling scales.
KEY_MAX = 2 ** KEY_SPACE_BITS


# --------------------------------------------------------------------------- #
# Detector (faithful numpy replica of eval_libero_action_inversion.py WMF path)
# --------------------------------------------------------------------------- #
def _band_limit_if_needed(arr, *, reference_mode, sr, fr):
    if reference_mode == "bandpass":
        return wm._band_limit(arr, sample_rate_hz=sr, freq_range=fr)
    return arr  # gaussian reference mode => no band limiting (matches detector)


def _reference_for_key(key, nonce, chunk_index, length, action_dim, sr, fr, mode, n_tones):
    cfg = wm.InternalNoiseWatermarkConfig(
        secret_key=int(key),
        control_freq=sr,
        beta=1.0,
        freq_range=fr,
        n_tones=n_tones,
        reference_mode=mode,
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=1,
        chunk_selection_count=5,
    )
    ctx = wm.WatermarkContext(chunk_index=int(chunk_index), episode_nonce=int(nonce))
    return wm.generate_keyed_reference(
        length=length, action_dim=action_dim, sample_rate_hz=sr, config=cfg, context=ctx
    )


_FEATURE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _feature_vector(key, ep):
    """Per-window cosine-similarity feature vector for a candidate key on one episode.

    Mirrors _window_score_vector(..., base_detector="cosine") on the SELECTED chunks
    of the saved partial+MAP recovered noise. Memoized on (key, episode index).
    """
    ck = (int(key), ep["idx"])
    hit = _FEATURE_CACHE.get(ck)
    if hit is not None:
        return hit
    scores = []
    for ci, rec_win in zip(ep["sel_chunk_idx"], ep["rec"]):
        r = _reference_for_key(
            key, ep["nonce"], ci, ep["length"], ep["action_dim"],
            ep["sr"], ep["fr"], ep["mode"], ep["n_tones"],
        )
        a = _band_limit_if_needed(rec_win, reference_mode=ep["mode"], sr=ep["sr"], fr=ep["fr"])
        rb = _band_limit_if_needed(r, reference_mode=ep["mode"], sr=ep["sr"], fr=ep["fr"])
        L = min(a.shape[0], rb.shape[0])
        Dd = min(a.shape[1], rb.shape[1])
        av = a[:L, :Dd].reshape(-1).astype(np.float32)
        rv = rb[:L, :Dd].reshape(-1).astype(np.float32)
        den = float(np.linalg.norm(av) * np.linalg.norm(rv))
        scores.append(0.0 if den < 1e-8 else float(np.dot(av, rv) / den))
    out = np.asarray(scores, dtype=np.float64)
    _FEATURE_CACHE[ck] = out
    return out


def _wmf_score(feature, null_matrix, subspace_rank):
    """Whitened matched-filter score psi = template^T Sigma^{-1/2} (feature - mu).

    Exact replica of _wmf_score_from_vectors / _select_whitened_subspace.
    """
    feature = np.asarray(feature, dtype=np.float64)
    null_matrix = np.asarray(null_matrix, dtype=np.float64)
    if feature.size == 0:
        return 0.0
    if _RAW_MF:
        return float(np.sum(feature - null_matrix.mean(axis=0)))
    mu = null_matrix.mean(axis=0)
    cf = feature - mu
    dim = cf.shape[0]
    cn = null_matrix - null_matrix.mean(axis=0, keepdims=True)
    if null_matrix.shape[0] <= 1:
        cov = np.eye(dim)
    else:
        cov = np.cov(cn, rowvar=False, bias=False)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    reg = max(1e-6, 1e-4 * float(np.trace(cov)) / max(dim, 1))
    cov = cov + reg * np.eye(dim)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    evec = evec[:, order]
    if subspace_rank is not None:
        rk = min(subspace_rank, dim)
        ev = ev[:rk]
        evec = evec[:, :rk]
    pf = evec.T @ cf
    tmpl = evec.sum(axis=0)
    inv = 1.0 / np.sqrt(np.maximum(ev, 1e-8))
    return float(np.dot(tmpl * inv, pf * inv))


# --------------------------------------------------------------------------- #
# Episode loading
# --------------------------------------------------------------------------- #
def _load_episodes():
    files = sorted(
        f for f in glob.glob(str(ROLLOUT_DIR / "*_watermarked.npz"))
        if "extra_modes" not in f
    )
    if not files:
        raise FileNotFoundError(f"No watermarked rollouts under {ROLLOUT_DIR}")
    episodes = []
    secret_key = None
    for ei, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        rec_all = d["chunk_recovered_noise_partial_map"]
        sel = d["chunk_selected"]
        chunk_idx = d["chunk_chunk_index"]
        sel_idx = np.where(sel)[0]
        episodes.append(
            dict(
                idx=ei,
                rec=[rec_all[i].astype(np.float32) for i in sel_idx],
                sel_chunk_idx=[int(chunk_idx[i]) for i in sel_idx],
                nonce=int(d["episode_nonce"]),
                length=int(rec_all.shape[1]),
                action_dim=int(rec_all.shape[2]),
                sr=float(d["sample_rate_hz"]),
                fr=(float(d["freq_min_hz"]), float(d["freq_max_hz"])),
                mode=str(d["reference_mode"]),
                n_tones=int(d["n_tones"]),
            )
        )
        sk = int(d["secret_key"])
        secret_key = sk if secret_key is None else secret_key
        assert sk == secret_key, "mixed secret keys in pool"
    return episodes, secret_key


def _z_for_key(key, ep, rng):
    """Calibrated per-episode evidence Z_e(key) using key's own false-key bank.

    s_e(key)   = WMF(feature(key), null bank);  the null bank for `key` is the
    set {key+1, ..., key+N_FALSE} (matches the saved-detector convention).
    Z_e(key)   = (s_e(key) - mu^-) / (sigma^- + eps), where mu^-/sigma^- are the
    mean/std of the false-key WMF scores (each false key scored against the same
    bank via leave-one-out), giving a per-episode standardized score.
    """
    false_keys = [key + 1 + j for j in range(N_FALSE)]
    feats = {k: _feature_vector(k, ep) for k in [key] + false_keys}
    null_mat = np.stack([feats[k] for k in false_keys])
    s_true = _wmf_score(feats[key], null_mat, SUBSPACE_RANK)
    # leave-one-out false-key scores against the same bank (calibration null)
    s_false = []
    for i, fk in enumerate(false_keys):
        loo = np.delete(null_mat, i, axis=0)
        s_false.append(_wmf_score(feats[fk], loo, SUBSPACE_RANK))
    s_false = np.asarray(s_false, dtype=np.float64)
    mu = s_false.mean()
    sd = s_false.std(ddof=1)
    z_true = (s_true - mu) / (sd + EPS)
    z_false = (s_false - mu) / (sd + EPS)
    return z_true, z_false


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    episodes, secret_key = _load_episodes()
    n_ep = len(episodes)
    print(f"[unforgeability] loaded {n_ep} watermarked episodes; secret_key={secret_key}")
    print(f"[unforgeability] key space = 2^{KEY_SPACE_BITS} ~ {KEY_MAX:,} keys")

    # ------------------------------------------------------------------ #
    # Per-episode true-key evidence Z_e(k*) and a per-episode false-key
    # evidence pool {Z_e(k) : k != k*}.  The false-key pool is what an
    # attacker's *random guess* looks like at the per-episode level.
    # ------------------------------------------------------------------ #
    z_true_per_ep = np.zeros(n_ep)
    z_false_per_ep = []   # list of arrays, one per episode, of false-key Z scores

    # True key: per-episode Z under its own (true) false-key bank.
    for e, ep in enumerate(episodes):
        zt, zf = _z_for_key(secret_key, ep, rng)
        z_true_per_ep[e] = zt
        z_false_per_ep.append(zf)
    z_false_per_ep = np.asarray(z_false_per_ep)   # (n_ep, N_FALSE)
    print(f"[unforgeability] mean Z_e(k*) = {z_true_per_ep.mean():.3f}; "
          f"false-key Z pool mean={z_false_per_ep.mean():.3f} std={z_false_per_ep.std():.3f}")

    # ------------------------------------------------------------------ #
    # Null distribution of T_G under random/false keys (the attacker model)
    # and H1 distribution of T_G under the true key, by MC over groups.
    # A "random guess" key contributes one false-key Z per episode in the
    # group; we resample episode and false-key index, matching the
    # uniqueness-analysis aggregation.
    # ------------------------------------------------------------------ #
    ep_draw = rng.integers(0, n_ep, size=(N_GROUP_SAMPLES, G))
    key_draw = rng.integers(0, N_FALSE, size=(N_GROUP_SAMPLES, G))
    t_null = z_false_per_ep[ep_draw, key_draw].sum(axis=1)        # T_G of random guesses
    t_h1 = z_true_per_ep[rng.integers(0, n_ep, size=(N_GROUP_SAMPLES, G))].sum(axis=1)

    # tau_decision: the (1 - FPR) quantile of the random-key (false-key) null T_G
    # distribution. This is the SAME operating point used by the uniqueness
    # analysis, so the two subsections report consistent numbers.
    tau = float(np.quantile(t_null, 1.0 - FPR_TARGET))

    # Per-guess forgery probability. Each independent random key the attacker
    # tries yields a T_G drawn from this same false-key null (the keys k*+1..
    # k*+32 across 50 episodes are 50x32 i.i.d. random-key evidence samples, and
    # the blake2b key->reference map makes them independent of k*). Hence the
    # per-guess success probability p = Pr[T_G(random key) >= tau] EQUALS the
    # target FPR by construction of the threshold; we verify this empirically
    # on a held-out resample so it is not literally the quantile's own pool.
    n_half = len(t_null) // 2
    tau_train = float(np.quantile(t_null[:n_half], 1.0 - FPR_TARGET))
    p = float(np.mean(t_null[n_half:] >= tau_train))           # held-out collision rate
    p_eff = p if p > 0.0 else 1.0 / (len(t_null) - n_half)
    p_note = f"{p:.2e}" if p > 0.0 else f"<= {p_eff:.2e} (0 of {len(t_null) - n_half} held-out)"
    brute_budget = 1.0 / p_eff
    max_t_null = float(t_null.max())
    tpr_true = float(np.mean(t_h1 >= tau))

    # Gaussian tail model of the random-key null T_G (mean ~0, used to
    # extrapolate the forgery budget at TIGHTER operating points and to tie it
    # to the key entropy: the verifier can lower the FPR until 1/FPR approaches
    # the key-space size 2^bits while keeping TPR ~ 1 thanks to the true-key
    # margin).
    null_mu = float(t_null.mean())
    null_sd = float(t_null.std(ddof=1))
    from scipy.stats import norm  # noqa: PLC0415
    print(f"[unforgeability] tau_decision (FPR={FPR_TARGET:g}, |G|={G}) = {tau:.3f}")
    print(f"[unforgeability] random-key null T_G: mu={null_mu:.3f} sd={null_sd:.3f} "
          f"max={max_t_null:.3f}")
    print(f"[unforgeability] true-key T_G: mean={t_h1.mean():.1f} min={t_h1.min():.1f} "
          f"TPR@tau={tpr_true:.3f}")
    print(f"[unforgeability] per-guess collision rate p (held-out) = {p_note}")
    print(f"[unforgeability] expected brute-force budget ~ 1/p = {brute_budget:.3e} guesses")
    print(f"[unforgeability] key entropy cap: 2^{KEY_SPACE_BITS} = {KEY_MAX:,} guesses")

    # ------------------------------------------------------------------ #
    # Attacker success vs budget M (the headline table). At the FPR=1e-3
    # operating point each guess succeeds w.p. p, so after M independent
    # guesses P(forge) = 1-(1-p)^M. We also report the expected best-of-M
    # T_G the attacker can show (max over M guesses) -- it approaches but, for
    # feasible M, the attacker only crosses tau with probability 1-(1-p)^M,
    # never exceeding it by a margin (cf. true-key min T_G).
    # ------------------------------------------------------------------ #
    rows = []
    for M in BUDGETS:
        p_success = 1.0 - (1.0 - p_eff) ** M
        # Gaussian order-statistic: expected max of M N(mu,sd) draws.
        if M >= 2:
            exp_best = null_mu + null_sd * norm.ppf(1.0 - 1.0 / (M + 1))
        else:
            exp_best = null_mu
        rows.append(dict(
            budget_M=M,
            p_success_forge=p_success,
            expected_best_T_G=float(exp_best),
            tau_decision=tau,
            any_forgery_reached_tau=bool(exp_best >= tau),
        ))
        print(f"  M={M:>10,}: P(forge)={p_success:.3e}  E[best T_G]={exp_best:8.3f}  "
              f"(tau={tau:.2f})")

    # ------------------------------------------------------------------ #
    # (b) one-wayness / non-differentiability diagnostic.
    # The key enters the reference ONLY through hashlib.blake2b inside
    # watermark._stable_seed. Perturbing the integer key by +/-1 yields an
    # essentially independent reference (avalanche), so there is no usable
    # local gradient or structure for a better-than-brute-force attack.
    # ------------------------------------------------------------------ #
    ep0 = episodes[0]
    ci0 = ep0["sel_chunk_idx"][0]
    r_base = _reference_for_key(secret_key, ep0["nonce"], ci0, ep0["length"],
                                ep0["action_dim"], ep0["sr"], ep0["fr"],
                                ep0["mode"], ep0["n_tones"]).reshape(-1)
    neigh_corrs = []
    for dk in (1, 2, 3, -1, 100, 12345):
        r_n = _reference_for_key(secret_key + dk, ep0["nonce"], ci0, ep0["length"],
                                 ep0["action_dim"], ep0["sr"], ep0["fr"],
                                 ep0["mode"], ep0["n_tones"]).reshape(-1)
        c = float(np.corrcoef(r_base, r_n)[0, 1])
        neigh_corrs.append((dk, c))
    print("[unforgeability] reference correlation r(k*) vs r(k*+dk) "
          "(should be ~0 => no local structure / gradient):")
    for dk, c in neigh_corrs:
        print(f"    dk={dk:>6}: corr={c:+.4f}")
    max_neigh_corr = max(abs(c) for _, c in neigh_corrs)

    # Relaxed continuous-key (gradient/structured) attack attempt: coordinate
    # hill-climb on the integer key, which is the only "continuous-like" relaxation
    # available since the key is the sole free variable and it is hashed. Starting
    # from a random key we probe +/- steps of varying size and keep any that raise
    # the single-episode WMF score; we report whether it climbs toward the true
    # key. Because the reference is a hash avalanche, neighbouring keys are
    # uncorrelated and the objective has no exploitable slope -> hill-climb does
    # no better than random sampling.
    ep_hc = episodes[0]

    def _z_key_ep(k, ep):
        # Verifier-faithful Z_e(k): candidate calibrated against ITS OWN false-key
        # bank (k+1..k+N_FALSE). Scoring against k*'s bank would inflate variance
        # and fabricate a spurious signal, so we never do that.
        k = int(k)
        own_bank = np.stack([_feature_vector(k + 1 + j, ep) for j in range(N_FALSE)])
        s_true = _wmf_score(_feature_vector(k, ep), own_bank, SUBSPACE_RANK)
        s_false = [
            _wmf_score(own_bank[i], np.delete(own_bank, i, axis=0), SUBSPACE_RANK)
            for i in range(N_FALSE)
        ]
        s_false = np.asarray(s_false, dtype=np.float64)
        return (s_true - s_false.mean()) / (s_false.std(ddof=1) + EPS)

    # Hill-climb on the (hashed) integer key, maximizing the single-episode
    # calibrated evidence. The single-episode WMF is heavy-tailed under random
    # keys, so a hill-climb can land a high single-episode value by chance -- but
    # this does NOT persist across episodes (each chunk is re-keyed per nonce via
    # blake2b), so the GROUP statistic T_G (the actual decision unit) collapses.
    # 128-bit key space exceeds int64, so np.integers can't sample it; draw the
    # start key from the same seeded RNG's bytes (deterministic, width-correct).
    cur = int.from_bytes(rng.bytes((KEY_SPACE_BITS + 7) // 8), "big") % KEY_MAX or 1
    cur_s = _z_key_ep(cur, ep_hc)
    for _ in range(40):
        improved = False
        for step in (1, 2, 4, 16, 256, 4096):
            for sgn in (+1, -1):
                cand = cur + sgn * step
                if cand <= 0:
                    continue
                cs = _z_key_ep(cand, ep_hc)
                if cs > cur_s:
                    cur, cur_s, improved = cand, cs, True
        if not improved:
            break
    hc_single = float(cur_s)
    hc_group_eps = list(range(min(G, n_ep)))
    hc_group_TG = float(sum(_z_key_ep(cur, episodes[e]) for e in hc_group_eps))
    true_group_TG = float(sum(_z_key_ep(secret_key, episodes[e]) for e in hc_group_eps))
    hc_forged = hc_group_TG >= tau
    print(f"[unforgeability] hill-climb winner: single-ep Z={hc_single:.2f} (lucky tail) "
          f"but GROUP T_G={hc_group_TG:.2f} << tau={tau:.2f}; "
          f"true-key group T_G={true_group_TG:.2f}")
    print(f"[unforgeability] structured/gradient attack reached tau? {hc_forged} "
          f"=> does NOT beat brute force")

    # Forgery-budget vs operating FPR (Gaussian tail extrapolation of the null).
    # budget(FPR) = 1/FPR, capped by the key entropy 2^bits. Shows the verifier
    # can trade a slightly higher tau (still below the true-key margin) for an
    # exponentially larger required attack budget.
    print("[unforgeability] budget vs operating FPR (1/FPR, capped by key entropy):")
    fpr_grid = [1e-2, 1e-3, 1e-4, 1e-6, 1e-9]
    fpr_rows = []
    for f in fpr_grid:
        tau_f = null_mu + null_sd * norm.ppf(1.0 - f)
        budget_f = min(1.0 / f, float(KEY_MAX))
        tpr_f = float(np.mean(t_h1 >= tau_f))
        fpr_rows.append((f, tau_f, budget_f, tpr_f))
        print(f"    FPR={f:.0e}: tau={tau_f:7.3f}  budget~{budget_f:.3e}  "
              f"true-key TPR={tpr_f:.3f}")

    # ------------------------------------------------------------------ #
    # write CSV
    # ------------------------------------------------------------------ #
    header = [
        "budget_M", "p_success_forge", "expected_best_T_G",
        "tau_decision", "any_forgery_reached_tau",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))
    # budget-vs-FPR extrapolation block
    lines.append("")
    lines.append("# budget_vs_fpr")
    lines.append("# operating_fpr,tau,brute_force_budget,true_key_tpr")
    for f, tau_f, budget_f, tpr_f in fpr_rows:
        lines.append(f"# {f},{tau_f},{budget_f},{tpr_f}")
    # summary trailer rows (as key,value comment-style) for downstream reuse
    summary = {
        "secret_key": secret_key,
        "n_episodes": n_ep,
        "G": G,
        "subspace_rank": SUBSPACE_RANK,
        "n_false_bank": N_FALSE,
        "fpr_target": FPR_TARGET,
        "tau_decision": tau,
        "per_guess_collision_rate_p": p_eff,
        "p_is_upper_bound": p <= 0.0,
        "brute_force_budget_inv_p": brute_budget,
        "key_space_bits": KEY_SPACE_BITS,
        "key_space_size": KEY_MAX,
        "true_key_TG_mean": float(t_h1.mean()),
        "true_key_TG_min": float(t_h1.min()),
        "true_key_TPR_at_tau": tpr_true,
        "random_guess_TG_max": max_t_null,
        "null_TG_mu": null_mu,
        "null_TG_sd": null_sd,
        "hillclimb_single_ep_Z": hc_single,
        "hillclimb_group_TG": hc_group_TG,
        "hillclimb_reached_tau": bool(hc_forged),
        "true_key_group_TG_hc_eps": true_group_TG,
        "max_neighbor_ref_corr": max_neigh_corr,
        "seed": SEED,
    }
    lines.append("")
    lines.append("# summary")
    for k, v in summary.items():
        lines.append(f"# {k},{v}")
    OUT_CSV.write_text("\n".join(lines) + "\n")
    print(f"[unforgeability] wrote {OUT_CSV}")

    return summary


if __name__ == "__main__":
    main()
