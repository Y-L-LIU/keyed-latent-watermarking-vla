#!/usr/bin/env python3
"""Run LIBERO action-inversion eval under output-controller postprocessing perturbations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from collections import deque
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclasses.dataclass(frozen=True)
class RobustnessConfig:
    controller_postprocess: str
    controller_clip_limit: float
    controller_smooth_alpha: float
    controller_jitter_std: float
    controller_delay_steps: int
    seed: int
    run_tag: str | None = None


class ControllerPostprocessor:
    def __init__(self, *, config: RobustnessConfig, action_dim: int):
        self.config = config
        self.action_dim = int(action_dim)
        self._episode_nonce: int | None = None
        self._smooth_prev = np.zeros((self.action_dim,), dtype=np.float32)
        self._delay_pending: deque[np.ndarray] = deque()
        self._delay_last_emitted: np.ndarray | None = None

    def _reset_episode(self, episode_nonce: int) -> None:
        self._episode_nonce = int(episode_nonce)
        self._smooth_prev = np.zeros((self.action_dim,), dtype=np.float32)
        self._delay_pending = deque()
        self._delay_last_emitted = None

    def _ensure_episode(self, episode_nonce: int) -> None:
        if self._episode_nonce != int(episode_nonce):
            self._reset_episode(episode_nonce)

    def apply_chunk(
        self,
        action_chunk: np.ndarray,
        *,
        episode_nonce: int,
        chunk_index: int,
    ) -> np.ndarray:
        self._ensure_episode(episode_nonce)
        actions = np.asarray(action_chunk, dtype=np.float32).copy()
        mode = self.config.controller_postprocess
        if mode == "none":
            return actions
        if mode == "clip":
            limit = float(self.config.controller_clip_limit)
            return np.clip(actions, -limit, limit).astype(np.float32)
        if mode == "smooth":
            alpha = float(self.config.controller_smooth_alpha)
            smoothed = np.zeros_like(actions, dtype=np.float32)
            prev = self._smooth_prev.astype(np.float32, copy=True)
            for idx, current in enumerate(actions):
                prev = (alpha * current) + ((1.0 - alpha) * prev)
                smoothed[idx] = prev
            self._smooth_prev = prev
            return smoothed
        if mode == "jitter":
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(self.config.seed),
                        int(episode_nonce) & 0xFFFFFFFF,
                        int(chunk_index) & 0xFFFFFFFF,
                    ]
                )
            )
            noise = rng.normal(loc=0.0, scale=float(self.config.controller_jitter_std), size=actions.shape)
            return (actions + noise.astype(np.float32)).astype(np.float32)
        if mode == "delay":
            delayed = np.zeros_like(actions, dtype=np.float32)
            delay_steps = max(0, int(self.config.controller_delay_steps))
            for idx, current in enumerate(actions):
                self._delay_pending.append(np.asarray(current, dtype=np.float32))
                if len(self._delay_pending) > delay_steps:
                    emitted = np.asarray(self._delay_pending.popleft(), dtype=np.float32)
                    self._delay_last_emitted = emitted
                    delayed[idx] = emitted
                else:
                    if self._delay_last_emitted is None:
                        self._delay_last_emitted = np.asarray(current, dtype=np.float32)
                    delayed[idx] = np.asarray(self._delay_last_emitted, dtype=np.float32)
            return delayed
        raise ValueError(f"Unsupported controller_postprocess={mode!r}")


def _parse_args(argv: list[str] | None = None) -> tuple[RobustnessConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--controller-postprocess",
        choices=("none", "clip", "smooth", "jitter", "delay"),
        default="none",
    )
    parser.add_argument("--controller-clip-limit", type=float, default=1.0)
    parser.add_argument("--controller-smooth-alpha", type=float, default=0.5)
    parser.add_argument("--controller-jitter-std", type=float, default=0.01)
    parser.add_argument("--controller-delay-steps", type=int, default=1)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--controller-seed", type=int, default=0)
    args, base_argv = parser.parse_known_args(argv)
    return (
        RobustnessConfig(
            controller_postprocess=args.controller_postprocess,
            controller_clip_limit=float(args.controller_clip_limit),
            controller_smooth_alpha=float(args.controller_smooth_alpha),
            controller_jitter_std=float(args.controller_jitter_std),
            controller_delay_steps=int(args.controller_delay_steps),
            seed=int(args.controller_seed),
            run_tag=args.run_tag,
        ),
        list(base_argv),
    )


def _default_run_tag(config: RobustnessConfig) -> str:
    mode = config.controller_postprocess
    if mode == "none":
        return "controller_none"
    if mode == "clip":
        return f"controller_clip_{config.controller_clip_limit:g}"
    if mode == "smooth":
        return f"controller_smooth_{config.controller_smooth_alpha:g}"
    if mode == "jitter":
        return f"controller_jitter_{config.controller_jitter_std:g}"
    if mode == "delay":
        return f"controller_delay_{config.controller_delay_steps}"
    raise ValueError(f"Unsupported controller_postprocess={mode!r}")


def _append_tag_to_base_argv(base_argv: list[str], run_tag: str) -> list[str]:
    tagged = list(base_argv)
    for option in ("--save-rollout-dir", "--save-report-dir"):
        prefix = f"{option}="
        for idx, token in enumerate(tagged):
            if token == option and idx + 1 < len(tagged):
                tagged[idx + 1] = str(pathlib.Path(tagged[idx + 1]) / run_tag)
                break
            if token.startswith(prefix):
                raw_path = token[len(prefix) :]
                tagged[idx] = f"{option}={pathlib.Path(raw_path) / run_tag}"
                break
    return tagged


def _write_metadata(root: pathlib.Path | None, *, config: RobustnessConfig, forwarded_argv: list[str]) -> None:
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "robustness_config.json").write_text(
        json.dumps(
            {
                "controller_postprocess": config.controller_postprocess,
                "controller_clip_limit": config.controller_clip_limit,
                "controller_smooth_alpha": config.controller_smooth_alpha,
                "controller_jitter_std": config.controller_jitter_std,
                "controller_delay_steps": config.controller_delay_steps,
                "seed": config.seed,
                "run_tag": config.run_tag,
                "forwarded_argv": forwarded_argv,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_base_module():
    from scripts import eval_libero_action_inversion as _base  # noqa: WPS433

    return _base


def main(argv: list[str] | None = None) -> int:
    _base = _load_base_module()
    config, base_argv = _parse_args(argv)
    run_tag = config.run_tag or _default_run_tag(config)
    forwarded_argv = _append_tag_to_base_argv(base_argv, run_tag=run_tag)
    base_args = _base._parse_args(forwarded_argv)
    _write_metadata(base_args.save_rollout_dir, config=config, forwarded_argv=forwarded_argv)
    _write_metadata(base_args.save_report_dir, config=config, forwarded_argv=forwarded_argv)

    original_sample_raw_actions = _base._sample_raw_actions
    controller_cache: dict[int, ControllerPostprocessor] = {}

    def _patched_sample_raw_actions(policy, obs: dict, *, noise: np.ndarray):
        outputs, injected_noise = original_sample_raw_actions(policy, obs, noise=noise)
        action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
        if config.controller_postprocess == "none":
            return outputs, injected_noise
        controller = controller_cache.get(int(action_chunk.shape[-1]))
        if controller is None:
            controller = ControllerPostprocessor(config=config, action_dim=int(action_chunk.shape[-1]))
            controller_cache[int(action_chunk.shape[-1])] = controller
        transformed_outputs = dict(outputs)
        transformed_outputs["actions"] = controller.apply_chunk(
            action_chunk,
            episode_nonce=int(obs.get("episode_nonce", -1)),
            chunk_index=int(obs.get("chunk_index", -1)),
        )
        return transformed_outputs, injected_noise

    _base._sample_raw_actions = _patched_sample_raw_actions
    try:
        return _base.main(forwarded_argv)
    finally:
        _base._sample_raw_actions = original_sample_raw_actions


if __name__ == "__main__":
    raise SystemExit(main())
