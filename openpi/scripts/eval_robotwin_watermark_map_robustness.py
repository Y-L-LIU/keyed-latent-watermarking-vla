#!/usr/bin/env python3
"""Run RoboTwin watermark MAP eval under output-controller postprocessing perturbations.

Four attack modes:
  - clip:   np.clip(actions, -limit, limit)
  - smooth: EMA with alpha
  - jitter: additive iid Gaussian noise
  - delay:  shift actions by N steps (hold-last during fill)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.eval_libero_action_inversion_postprocess_robustness import (
    ControllerPostprocessor,
    RobustnessConfig,
    _default_run_tag,
)
from scripts import eval_robotwin_watermark_map as _base


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


def _append_output_tag(base_argv: list[str], run_tag: str) -> list[str]:
    tagged = list(base_argv)
    for idx, token in enumerate(tagged):
        if token == "--output-dir" and idx + 1 < len(tagged):
            tagged[idx + 1] = str(pathlib.Path(tagged[idx + 1]) / run_tag)
            break
        if token.startswith("--output-dir="):
            raw_path = token[len("--output-dir="):]
            tagged[idx] = f"--output-dir={pathlib.Path(raw_path) / run_tag}"
            break
    else:
        pass
    return tagged


def _write_metadata(output_dir: pathlib.Path | None, *, config: RobustnessConfig, forwarded_argv: list[str]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness_config.json").write_text(
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


def main(argv: list[str] | None = None) -> int:
    config, base_argv = _parse_args(argv)
    run_tag = config.run_tag or _default_run_tag(config)
    forwarded_argv = _append_output_tag(base_argv, run_tag=run_tag)

    # Write metadata to the output directory (parse --output-dir from forwarded_argv)
    output_dir = None
    for idx, token in enumerate(forwarded_argv):
        if token == "--output-dir" and idx + 1 < len(forwarded_argv):
            output_dir = pathlib.Path(forwarded_argv[idx + 1])
            break
        if token.startswith("--output-dir="):
            output_dir = pathlib.Path(token[len("--output-dir="):])
            break
    _write_metadata(output_dir, config=config, forwarded_argv=forwarded_argv)

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
        # eval_robotwin_watermark_map.main() reads from sys.argv, so patch it
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + forwarded_argv
        try:
            _base.main()
        finally:
            sys.argv = old_argv
    finally:
        _base._sample_raw_actions = original_sample_raw_actions

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
