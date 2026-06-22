#!/usr/bin/env python3
"""Build a 'compressed' suspect checkpoint by magnitude pruning or int8 quantization
applied to weight tensors of an openpi (pi0/pi0.5) descendant checkpoint. Saves a new
ckpt dir that the existing eval scripts can consume via --checkpoint-dir.

--scope all (default): prune/quant the whole model (VLM backbone + action expert).
--scope action: prune/quant ONLY the action policy (the action expert — gemma weights
  with a numeric suffix like 'attn_1'/'mlp_1' — plus the action projection heads), leaving
  the VLM backbone (SigLIP vision + Gemma-2B prefix expert) bit-for-bit untouched. This is
  the more targeted threat model: compress the component that generates the watermarked
  actions, not the perception/instruction backbone.

The detector is NOT modified — the verifier keeps the original base ckpt.
This realizes the §12.5 (pruning / quantization) row of the threat model:
deploy a compressed suspect, verify with the unmodified detector.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from openpi.models import model as _model


def _path_str(path):
    parts = []
    for k in path:
        if hasattr(k, "key"):
            parts.append(str(k.key))
        elif hasattr(k, "name"):
            parts.append(str(k.name))
        else:
            parts.append(str(k))
    return "/".join(parts)


def mag_prune(arr: np.ndarray, sparsity: float) -> np.ndarray:
    orig_dtype = arr.dtype
    a = np.asarray(arr).astype(np.float32, copy=False)
    if sparsity <= 0.0:
        return arr
    k = int(a.size * sparsity)
    if k <= 0:
        return arr
    flat_abs = np.abs(a).reshape(-1)
    thresh = np.partition(flat_abs, k - 1)[k - 1]
    mask = (np.abs(a) > thresh)
    out = np.where(mask, a, 0.0)
    return out.astype(orig_dtype, copy=False)


def int8_quant(arr: np.ndarray, per_channel_axis: int | None = -1) -> np.ndarray:
    """Per-output-channel symmetric int8 fake-quant (common deployment recipe).
    Quantize-dequantize so the eval pipeline stays unchanged (weights live in
    bf16 but only take 256 distinct values per output channel)."""
    orig_dtype = arr.dtype
    a = np.asarray(arr).astype(np.float32, copy=False)
    if per_channel_axis is None or a.ndim < 2:
        scale = max(float(np.max(np.abs(a))), 1e-8) / 127.0
        q = np.round(a / scale).clip(-127, 127)
        deq = q * scale
    else:
        axis = per_channel_axis if per_channel_axis >= 0 else a.ndim + per_channel_axis
        # max over all axes except `axis`
        reduce_axes = tuple(i for i in range(a.ndim) if i != axis)
        amax = np.max(np.abs(a), axis=reduce_axes, keepdims=True)
        scale = np.maximum(amax, 1e-8) / 127.0
        q = np.round(a / scale).clip(-127, 127)
        deq = q * scale
    return deq.astype(orig_dtype, copy=False)


_WEIGHT_LEAF_NAMES = {
    # pi05 vision (PaliGemma img) + action heads + time-mlp + post-norm Dense
    "kernel",
    # pi05 LLM (gemma) base einsum weights
    "w",
    "gating_einsum",
    "linear",
    # LoRA adapters added during descendant fine-tune
    "lora_a",
    "lora_b",
    "gating_einsum_lora_a",
    "gating_einsum_lora_b",
    "linear_lora_a",
    "linear_lora_b",
}
# Skip: bias, scale (LN), pos_embedding (learned positional), input_embedding (token table)

# Action-policy submodules, for --scope action (prune ONLY the action policy, leave the
# VLM backbone — SigLIP vision + Gemma-2B prefix expert — untouched). In pi0/pi0.5 the
# action expert is the second gemma expert: its weights carry a numeric suffix (e.g.
# "attn_1", "mlp_1", "final_norm_1") while the VLM/prefix expert (index 0) has no suffix
# (gemma.py:_name). The action projection heads live outside gemma at the Pi0 top level.
_ACTION_HEAD_MODULES = {
    "action_in_proj", "action_out_proj", "state_proj",
    "time_mlp_in", "time_mlp_out", "action_time_mlp_in", "action_time_mlp_out",
}
_EXPERT_SUFFIX_RE = re.compile(r"_[1-9][0-9]*$")  # expert index >= 1 (action expert); _0/no-suffix = VLM


def _comp(k):
    return k.key if hasattr(k, "key") else (k.name if hasattr(k, "name") else str(k))


def is_action_policy(path) -> bool:
    comps = [_comp(k) for k in path]
    # action projection heads (outside the gemma LLM)
    if any(c in _ACTION_HEAD_MODULES for c in comps):
        return True
    # action expert inside the LLM: a suffixed (_1, _2, ...) submodule under "llm".
    # The "llm" guard prevents matching SigLIP's Dense_1 etc. (those live under "img").
    if "llm" in comps and any(_EXPERT_SUFFIX_RE.search(c) for c in comps):
        return True
    return False


def attack_leaf(path, leaf, attack: str, prune_sparsity: float, scope: str = "all"):
    name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
    if name not in _WEIGHT_LEAF_NAMES or leaf.ndim < 2:
        return leaf, False
    if scope == "action" and not is_action_policy(path):
        return leaf, False
    if attack == "prune":
        out = mag_prune(np.asarray(leaf), prune_sparsity)
    elif attack == "quant":
        out = int8_quant(np.asarray(leaf), per_channel_axis=-1)
    else:
        raise ValueError(attack)
    return out, True


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    os.symlink(src.resolve(), dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", required=True, help="src openpi step dir (contains params/, assets/)")
    ap.add_argument("--dst-ckpt", required=True, help="dst dir to write attacked ckpt")
    ap.add_argument("--attack", choices=["prune", "quant"], required=True)
    ap.add_argument("--prune-sparsity", type=float, default=0.3)
    ap.add_argument(
        "--scope",
        choices=["all", "action"],
        default="all",
        help="all = whole model (default, backward-compatible); "
        "action = only the action policy (action expert + projection heads), "
        "leaving the VLM backbone (SigLIP vision + Gemma-2B prefix expert) untouched.",
    )
    args = ap.parse_args()

    src = pathlib.Path(args.src_ckpt).resolve()
    dst = pathlib.Path(args.dst_ckpt).resolve()
    if not (src / "params").exists():
        print(f"ERR: {src}/params missing")
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    # Symlink assets and _CHECKPOINT_METADATA from the source (they are not modified)
    for sub in ("assets", "_CHECKPOINT_METADATA"):
        if (src / sub).exists():
            _link_or_copy(src / sub, dst / sub)
    # Remove any existing params/
    params_dst = dst / "params"
    if params_dst.is_symlink():
        params_dst.unlink()
    elif params_dst.exists():
        shutil.rmtree(params_dst)

    # Load source params on CPU as numpy
    print(f"[{args.attack}] loading params from {src/'params'} ...", flush=True)
    params = _model.restore_params(src / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)

    # Apply attack
    stats = {"num_leaves": 0, "num_attacked": 0, "total_params_attacked": 0}

    def _apply(path, leaf):
        stats["num_leaves"] += 1
        new_leaf, attacked = attack_leaf(path, leaf, args.attack, args.prune_sparsity, args.scope)
        if attacked:
            stats["num_attacked"] += 1
            stats["total_params_attacked"] += int(new_leaf.size)
        return new_leaf

    print(f"[{args.attack}] applying ...", flush=True)
    new_params = jax.tree_util.tree_map_with_path(_apply, params)

    # Save to dst/params via orbax PyTreeCheckpointer
    print(f"[{args.attack}] saving to {params_dst} ...", flush=True)
    save_args = jax.tree.map(lambda _: ocp.SaveArgs(), new_params)
    with ocp.PyTreeCheckpointer() as ck:
        ck.save(
            params_dst,
            args=ocp.args.PyTreeSave(item={"params": new_params}, save_args={"params": save_args}),
        )

    stats_path = dst / "attack_stats.json"
    stats["attack"] = args.attack
    stats["scope"] = args.scope
    stats["prune_sparsity"] = args.prune_sparsity if args.attack == "prune" else None
    stats["src_ckpt"] = str(src)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[{args.attack}] done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
