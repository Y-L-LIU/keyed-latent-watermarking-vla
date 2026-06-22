"""Shared DC (temporally-persistent) keyed-offset helpers.

Used identically by the relabel (injection) and the bias-retention test (detection),
so the formulas cannot drift. A constant-in-time per-task offset survives the BC
data loader's all-phase slicing (unlike a zero-mean temporal reference) and is the
maximally-learnable mark -> the positive control for distillation survival.
"""
import hashlib
import numpy as np


def prompt_seed(prompt) -> int:
    return int.from_bytes(hashlib.blake2b(str(prompt).encode("utf-8"), digest_size=8).digest(), "little")


def dc_offset(secret_key: int, task_seed: int, action_dim: int) -> np.ndarray:
    """Fixed (non-zero-mean) per-(key,task) action-space offset vector, shape (action_dim,)."""
    s = int.from_bytes(
        hashlib.blake2b(f"{int(secret_key)}:{int(task_seed)}".encode("utf-8"), digest_size=8).digest(),
        "little",
    ) % (2**32)
    return np.random.default_rng(s).standard_normal(action_dim).astype(np.float64)


def dc_bias(secret_key: int, prompt, horizon: int, action_dim: int, beta_out: float) -> np.ndarray:
    """beta_out * offset, tiled over the horizon -> shape (horizon, action_dim)."""
    c = dc_offset(secret_key, prompt_seed(prompt), action_dim)
    return float(beta_out) * np.tile(c[None, :], (horizon, 1))
