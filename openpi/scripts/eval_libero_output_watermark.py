#!/usr/bin/env python3
"""Evaluate action-space (output) sinusoidal watermark baseline on LIBERO.

Adds a deterministic sine-wave reference directly to the policy's output actions,
then detects it via matched filter on the raw model output. This serves as a
simple baseline for comparison with the internal-noise latent watermark.

Reference for each (chunk, dim):
    r(t) = sin(2 * pi * freq_d * (t / sample_rate) + phase_d)
where freq_d and phase_d are derived from (secret_key, chunk_index, episode_nonce, dim).

Injection:  action_out = action + beta * r
Detection:  matched_filter_score = |<trace, r>| / ||r||   (per-dim, then averaged)
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import logging
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_libero_internal_watermark as online_eval


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=str, default="pi05_libero")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--task-suite-name", type=str, default="libero_10")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--num-trials-per-task", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    # Watermark params
    parser.add_argument("--secret-key", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--freq-min-hz", type=float, default=1.0)
    parser.add_argument("--freq-max-hz", type=float, default=2.0)
    # Output
    parser.add_argument("--save-dir", type=pathlib.Path, default=None)
    parser.add_argument("--smooth-alpha", type=float, default=None,
                        help="If set, apply EMA smoothing to actions before sending to env (attack simulation)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Sine-wave reference generation
# ---------------------------------------------------------------------------

def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _generate_sine_reference(
    *,
    length: int,
    action_dim: int,
    sample_rate_hz: float,
    secret_key: int,
    episode_nonce: int,
    freq_range: tuple[float, float],
    time_offset: int = 0,
) -> np.ndarray:
    """Generate a (length, action_dim) sine-wave reference.

    Each dimension gets a deterministic frequency in [freq_min, freq_max] and a
    random phase in [0, 2*pi], both derived from the secret key and episode nonce
    (NOT chunk_index, so the wave is continuous across replan boundaries).
    time_offset is the global step index at which this segment starts.
    """
    f_min, f_max = freq_range
    t = (np.arange(length, dtype=np.float64) + time_offset) / sample_rate_hz
    ref = np.zeros((length, action_dim), dtype=np.float32)
    for dim in range(action_dim):
        seed = _stable_seed(secret_key, episode_nonce, dim)
        rng = np.random.default_rng(seed)
        freq = rng.uniform(f_min, f_max)
        phase = rng.uniform(0.0, 2 * np.pi)
        ref[:, dim] = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    return ref


# ---------------------------------------------------------------------------
# Matched-filter detection (per-dim, mean-centered)
# ---------------------------------------------------------------------------

def _matched_filter_score(
    trace: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Per-dimension |<trace, ref>| / ||ref||, averaged across dims."""
    assert trace.shape == reference.shape
    scores = []
    for dim in range(trace.shape[1]):
        x = trace[:, dim].astype(np.float64) - np.mean(trace[:, dim])
        r = reference[:, dim].astype(np.float64) - np.mean(reference[:, dim])
        norm_r = np.linalg.norm(r)
        if norm_r < 1e-8:
            scores.append(0.0)
            continue
        scores.append(float(abs(np.dot(x, r)) / norm_r))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    policy,
    *,
    task,
    initial_state: np.ndarray,
    args: argparse.Namespace,
    runtime_modules: dict,
    episode_nonce: int,
    add_watermark: bool,
) -> dict:
    """Run a single LIBERO episode, optionally adding sine watermark."""
    env, task_description = online_eval._get_libero_env(
        task,
        resolution=args.resize_size,
        seed=args.seed,
    )
    max_steps = online_eval._suite_max_steps(args.task_suite_name)

    action_plan: collections.deque = collections.deque()
    action_traces: list[np.ndarray] = []
    chunk_index = 0
    global_action_step = 0
    done = False
    t = 0
    smooth_prev = None

    try:
        env.reset()
        obs = env.set_init_state(copy.deepcopy(initial_state))

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(online_eval.LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if not action_plan:
                element = online_eval._prepare_policy_observation(
                    obs,
                    task_description=task_description,
                    resize_size=args.resize_size,
                    image_tools=runtime_modules["image_tools"],
                )
                outputs = policy.infer(element)
                action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
                planned_steps = min(args.replan_steps, action_chunk.shape[0])

                if add_watermark:
                    ref = _generate_sine_reference(
                        length=planned_steps,
                        action_dim=action_chunk.shape[1],
                        sample_rate_hz=args.sample_rate_hz,
                        secret_key=args.secret_key,
                        episode_nonce=episode_nonce,
                        freq_range=(args.freq_min_hz, args.freq_max_hz),
                        time_offset=global_action_step,
                    )
                    action_chunk[:planned_steps] = action_chunk[:planned_steps] + args.beta * ref

                # Apply EMA smoothing (simulates downstream controller attack)
                if args.smooth_alpha is not None:
                    alpha = args.smooth_alpha
                    for i in range(planned_steps):
                        if smooth_prev is None:
                            smooth_prev = action_chunk[i].copy()
                        else:
                            smooth_prev = alpha * action_chunk[i] + (1 - alpha) * smooth_prev
                        action_chunk[i] = smooth_prev

                action_traces.append(action_chunk[:planned_steps].copy())
                action_plan.extend(action_chunk[:planned_steps])
                chunk_index += 1
                global_action_step += planned_steps

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
            t += 1

    finally:
        env.close()

    return {
        "success": bool(done),
        "action_traces": action_traces,
        "total_steps": t,
        "num_chunks": chunk_index,
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_episode(
    action_traces: list[np.ndarray],
    *,
    episode_nonce: int,
    secret_key: int,
    sample_rate_hz: float,
    freq_range: tuple[float, float],
) -> float:
    """Compute matched-filter score on concatenated episode trace.

    The reference is a single continuous sine wave over the full episode
    duration, matching the injection.
    """
    if not action_traces:
        return 0.0
    full_trace = np.concatenate(action_traces, axis=0)
    action_dim = full_trace.shape[1]
    total_length = full_trace.shape[0]
    full_ref = _generate_sine_reference(
        length=total_length,
        action_dim=action_dim,
        sample_rate_hz=sample_rate_hz,
        secret_key=secret_key,
        episode_nonce=episode_nonce,
        freq_range=freq_range,
        time_offset=0,
    )
    return _matched_filter_score(full_trace, full_ref)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    runtime_modules = online_eval._load_runtime_modules()
    train_config = runtime_modules["training_config"].get_config(args.config_name)

    benchmark_dict = runtime_modules["benchmark"].get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_start = int(args.task_offset)
    task_stop = min(task_suite.n_tasks, task_start + args.num_tasks)

    freq_range = (args.freq_min_hz, args.freq_max_hz)
    wrong_key = args.secret_key + 9999

    plain_scores: list[float] = []
    marked_scores: list[float] = []
    wrong_key_scores: list[float] = []
    plain_successes = 0
    marked_successes = 0
    total_episodes = 0

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(task_start, task_stop):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        num_trials = min(args.num_trials_per_task, len(initial_states))
        task_description = task.language
        logging.info("Task %d: %s (%d trials)", task_id, task_description, num_trials)

        policy = runtime_modules["policy_config"].create_trained_policy(
            train_config, args.checkpoint_dir
        )
        try:
            for episode_idx in range(num_trials):
                episode_nonce = (task_id + 1) * 100_000 + episode_idx
                initial_state = initial_states[episode_idx]

                plain_result = _run_episode(
                    policy,
                    task=task,
                    initial_state=initial_state,
                    args=args,
                    runtime_modules=runtime_modules,
                    episode_nonce=episode_nonce,
                    add_watermark=False,
                )

                marked_result = _run_episode(
                    policy,
                    task=task,
                    initial_state=initial_state,
                    args=args,
                    runtime_modules=runtime_modules,
                    episode_nonce=episode_nonce,
                    add_watermark=True,
                )

                p_score = _detect_episode(
                    plain_result["action_traces"],
                    episode_nonce=episode_nonce,
                    secret_key=args.secret_key,
                    sample_rate_hz=args.sample_rate_hz,
                    freq_range=freq_range,
                )
                m_score = _detect_episode(
                    marked_result["action_traces"],
                    episode_nonce=episode_nonce,
                    secret_key=args.secret_key,
                    sample_rate_hz=args.sample_rate_hz,
                    freq_range=freq_range,
                )
                wk_score = _detect_episode(
                    marked_result["action_traces"],
                    episode_nonce=episode_nonce,
                    secret_key=wrong_key,
                    sample_rate_hz=args.sample_rate_hz,
                    freq_range=freq_range,
                )

                plain_scores.append(p_score)
                marked_scores.append(m_score)
                wrong_key_scores.append(wk_score)

                if plain_result["success"]:
                    plain_successes += 1
                if marked_result["success"]:
                    marked_successes += 1
                total_episodes += 1

                logging.info(
                    "  ep%d: plain=%.3f wm=%.3f wk=%.3f | success: plain=%s wm=%s",
                    episode_idx,
                    p_score,
                    m_score,
                    wk_score,
                    plain_result["success"],
                    marked_result["success"],
                )

                if args.save_dir is not None:
                    np.savez_compressed(
                        args.save_dir / f"episode_t{task_id}_e{episode_idx}.npz",
                        task_id=task_id,
                        episode_idx=episode_idx,
                        episode_nonce=episode_nonce,
                        plain_score=p_score,
                        marked_score=m_score,
                        wrong_key_score=wk_score,
                        plain_success=plain_result["success"],
                        marked_success=marked_result["success"],
                        beta=args.beta,
                        secret_key=args.secret_key,
                    )

        finally:
            del policy

    # Summary
    p = np.array(plain_scores)
    m = np.array(marked_scores)
    wk = np.array(wrong_key_scores)

    print("\n" + "=" * 70)
    print("OUTPUT ACTION WATERMARK BASELINE RESULTS")
    print("=" * 70)
    print(f"Episodes: {total_episodes} | beta={args.beta} | freq=[{args.freq_min_hz}, {args.freq_max_hz}] Hz")
    print(f"Success rate: plain={plain_successes}/{total_episodes} "
          f"({100*plain_successes/max(total_episodes,1):.1f}%) | "
          f"wm={marked_successes}/{total_episodes} "
          f"({100*marked_successes/max(total_episodes,1):.1f}%)")
    print()
    print(f"Matched-filter scores:")
    print(f"  Plain:     mean={p.mean():.4f}  std={p.std():.4f}  min={p.min():.4f}  max={p.max():.4f}")
    print(f"  Marked:    mean={m.mean():.4f}  std={m.std():.4f}  min={m.min():.4f}  max={m.max():.4f}")
    print(f"  Wrong-key: mean={wk.mean():.4f}  std={wk.std():.4f}  min={wk.min():.4f}  max={wk.max():.4f}")
    print()

    auc = online_eval._roc_auc(marked_scores, plain_scores)
    print(f"AUC (marked vs plain): {auc:.4f}")
    print(f"  min(marked)={m.min():.4f}  max(plain)={p.max():.4f}  gap={m.min() - p.max():.4f}")

    if args.save_dir is not None:
        summary = {
            "total_episodes": total_episodes,
            "beta": args.beta,
            "secret_key": args.secret_key,
            "freq_range": [args.freq_min_hz, args.freq_max_hz],
            "sample_rate_hz": args.sample_rate_hz,
            "auc": auc,
            "plain_success_rate": plain_successes / max(total_episodes, 1),
            "marked_success_rate": marked_successes / max(total_episodes, 1),
            "plain_score_mean": float(p.mean()),
            "marked_score_mean": float(m.mean()),
            "wrong_key_score_mean": float(wk.mean()),
        }
        (args.save_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        logging.info("Results saved to %s", args.save_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
