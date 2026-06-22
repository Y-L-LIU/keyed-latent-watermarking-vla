"""Lightweight LoRA for the LingBot-VA Wan transformer.

LingBot-VA's trainer (wan_va/train.py) only supports full-parameter FSDP fine-tuning.
This module adds parameter-efficient LoRA without any external dependency (peft is not
in the env). It is designed so that:

  * LoRA adapters are injected into the attention projections (to_q/to_k/to_v/to_out)
    and optionally the FFN linears of every transformer block, BEFORE FSDP sharding.
  * Only the LoRA params train; the base weights stay frozen.
  * At save time, the gathered (full, un-sharded) state dict is folded back into plain
    Linear weights via `merge_lora_state_dict`, so the checkpoint loads with the stock
    `WanTransformer3DModel.from_pretrained` used by the eval/server path — i.e. a LoRA
    descendant is indistinguishable from a normally fine-tuned checkpoint at load time.

State-dict layout for a wrapped linear at module path `<p>`:
    <p>.base.weight, <p>.base.bias       (frozen original)
    <p>.lora_A.weight                    (trainable, [rank, in])
    <p>.lora_B.weight                    (trainable, [out, rank])
    <p>.lora_scaling                     (buffer, scalar = alpha / rank)
`merge_lora_state_dict` rewrites these into `<p>.weight` / `<p>.bias`.
"""

import math

import torch
import torch.nn as nn

# Canonical LoRA targets inside a WanAttention module.
DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out")


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update: y = base(x) + scaling * B(A(x))."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(alpha)
        scaling = self.alpha / self.rank

        w = base.weight
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # B=0 -> adapter starts as identity
        self.lora_A.to(device=w.device, dtype=w.dtype)
        self.lora_B.to(device=w.device, dtype=w.dtype)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.register_buffer("lora_scaling", torch.tensor(scaling, dtype=w.dtype, device=w.device))

    def forward(self, x):
        return self.base(x) + self.lora_scaling * self.lora_B(self.lora_A(self.dropout(x)))


def inject_lora(model, rank=16, alpha=16, dropout=0.0, targets=DEFAULT_TARGETS, include_ffn=False):
    """Replace target Linear submodules in every transformer block with LoRALinear.

    Returns the number of wrapped linears. Call this on the freshly loaded (CPU) model,
    before activation checkpointing and FSDP sharding.
    """
    targets = tuple(targets)
    n = 0
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no `.blocks`; cannot inject LoRA")
    for block in blocks:
        for attn_name in ("attn1", "attn2"):
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for t in targets:
                if t == "to_out":
                    mod = getattr(attn, "to_out", None)
                    if mod is not None and len(mod) > 0 and isinstance(mod[0], nn.Linear):
                        mod[0] = LoRALinear(mod[0], rank, alpha, dropout)
                        n += 1
                else:
                    lin = getattr(attn, t, None)
                    if isinstance(lin, nn.Linear):
                        setattr(attn, t, LoRALinear(lin, rank, alpha, dropout))
                        n += 1
        if include_ffn and hasattr(block, "ffn"):
            net = getattr(block.ffn, "net", None)
            if net is not None:
                for i, m in enumerate(net):
                    if isinstance(m, nn.Linear):
                        net[i] = LoRALinear(m, rank, alpha, dropout)
                        n += 1
                    elif hasattr(m, "proj") and isinstance(m.proj, nn.Linear):
                        m.proj = LoRALinear(m.proj, rank, alpha, dropout)
                        n += 1
    return n


def mark_only_lora_trainable(model):
    """Freeze everything except LoRA params. Returns (trainable, total) param counts."""
    trainable = total = 0
    for name, p in model.named_parameters():
        is_lora = (".lora_A." in name) or (".lora_B." in name)
        p.requires_grad_(is_lora)
        total += p.numel()
        if is_lora:
            trainable += p.numel()
    return trainable, total


@torch.no_grad()
def merge_lora_state_dict(state_dict):
    """Fold LoRALinear entries in a full state dict back into plain Linear weights.

    Produces keys compatible with the stock WanTransformer3DModel. Non-LoRA keys pass
    through unchanged.
    """
    prefixes = [k[: -len(".lora_A.weight")] for k in state_dict if k.endswith(".lora_A.weight")]
    pset = set(prefixes)
    drop = set()
    for p in prefixes:
        drop.update({p + ".lora_A.weight", p + ".lora_B.weight", p + ".lora_scaling"})

    out = {}
    for k, v in state_dict.items():
        if k in drop:
            continue
        # rewrite "<p>.base.weight/bias" -> "<p>.weight/bias", merging in the low-rank delta
        matched = None
        for p in pset:
            if k == p + ".base.weight" or k == p + ".base.bias":
                matched = p
                break
        if matched is None:
            out[k] = v
            continue
        if k.endswith(".base.bias"):
            out[matched + ".bias"] = v
        else:  # base.weight
            A = state_dict[matched + ".lora_A.weight"]
            B = state_dict[matched + ".lora_B.weight"]
            scaling = state_dict.get(matched + ".lora_scaling")
            s = float(scaling) if scaling is not None else 1.0
            delta = (B.to(torch.float32) @ A.to(torch.float32)) * s
            out[matched + ".weight"] = (v.to(torch.float32) + delta).to(v.dtype)
    return out
