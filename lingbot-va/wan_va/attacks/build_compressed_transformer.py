#!/usr/bin/env python3
"""Apply magnitude pruning or per-channel int8 fake-quant to the Wan transformer
weights of a lingbot LoRA-merged descendant checkpoint. Writes a new transformer/
dir (config.json + diffusion_pytorch_model.safetensors) that the existing eval
pipeline picks up.

The detector (owner-side) stays at the original base transformer — same protocol
as openpi: pruning/quantization is applied only on the suspect (deployed) side.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file


# tensors whose name ends with one of these get attacked (skip biases, 1-D norms,
# and any leaf with ndim < 2). Lingbot Wan transformer uses the standard
# `.weight` / `.bias` Linear naming.
_WEIGHT_SUFFIXES = (".weight",)
# Optional skip list (substring match) — these were never trained as Linears.
_SKIP_SUBSTRINGS: tuple[str, ...] = ()


def mag_prune(t: torch.Tensor, sparsity: float) -> torch.Tensor:
    orig = t.dtype
    a = t.detach().to(torch.float32)
    if sparsity <= 0.0 or a.numel() == 0:
        return t
    k = int(a.numel() * sparsity)
    if k <= 0:
        return t
    flat_abs = a.abs().reshape(-1)
    # use topk on the values to be pruned (kth smallest |w|) — equivalent to threshold
    thresh = torch.kthvalue(flat_abs, k).values
    mask = a.abs() > thresh
    return torch.where(mask, a, torch.zeros_like(a)).to(orig)


def int8_quant(t: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 fake-quant. Channel = dim 0 for Linear
    `.weight` (out_features, in_features). For higher-rank kernels, channel = dim 0.
    """
    orig = t.dtype
    a = t.detach().to(torch.float32)
    if a.ndim < 2:
        return t
    reduce_axes = tuple(range(1, a.ndim))
    amax = a.abs().amax(dim=reduce_axes, keepdim=True)
    scale = torch.clamp(amax, min=1e-8) / 127.0
    q = torch.round(a / scale).clamp(-127, 127)
    return (q * scale).to(orig)


def should_attack(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    if not name.endswith(_WEIGHT_SUFFIXES):
        return False
    for sub in _SKIP_SUBSTRINGS:
        if sub in name:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-transformer", required=True, help="src .../transformer/ dir")
    ap.add_argument("--dst-transformer", required=True, help="dst .../transformer/ dir")
    ap.add_argument("--attack", choices=["prune", "quant"], required=True)
    ap.add_argument("--prune-sparsity", type=float, default=0.3)
    args = ap.parse_args()

    src = pathlib.Path(args.src_transformer).resolve()
    dst = pathlib.Path(args.dst_transformer).resolve()
    src_st = src / "diffusion_pytorch_model.safetensors"
    src_index = src / "diffusion_pytorch_model.safetensors.index.json"
    if not src_st.exists() and not src_index.exists():
        print(f"ERR: no single-file or sharded safetensors under {src}")
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    # copy config.json verbatim
    if (src / "config.json").exists():
        shutil.copy2(src / "config.json", dst / "config.json")

    # load single-file OR sharded safetensors (posttrain base is sharded)
    if src_st.exists():
        print(f"[{args.attack}] loading {src_st} ...", flush=True)
        sd = load_file(str(src_st))
    else:
        weight_map = json.loads(src_index.read_text())["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        print(f"[{args.attack}] loading {len(shard_files)} shards from {src_index} ...", flush=True)
        sd = {}
        for sf in shard_files:
            sd.update(load_file(str(src / sf)))
    stats = {"num_tensors": len(sd), "num_attacked": 0, "params_attacked": 0}
    out: dict[str, torch.Tensor] = {}
    for name, t in sd.items():
        if should_attack(name, t):
            if args.attack == "prune":
                t = mag_prune(t, args.prune_sparsity)
            else:
                t = int8_quant(t)
            stats["num_attacked"] += 1
            stats["params_attacked"] += int(t.numel())
        out[name] = t
    dst_st = dst / "diffusion_pytorch_model.safetensors"
    print(f"[{args.attack}] writing {dst_st} ...", flush=True)
    save_file(out, str(dst_st), metadata={"format": "pt"})
    stats_path = dst / "attack_stats.json"
    stats["attack"] = args.attack
    stats["prune_sparsity"] = args.prune_sparsity if args.attack == "prune" else None
    stats["src_transformer"] = str(src)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[{args.attack}] done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
