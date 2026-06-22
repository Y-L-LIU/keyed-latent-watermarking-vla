"""Assemble a loadable LingBot model dir from a base model + a (LoRA-merged) trained transformer.

LingBot loads a model dir with {vae, text_encoder, tokenizer, transformer, ...}. After a LoRA
fine-tune, only the transformer changed. To use a stage-1 descendant as the BASE for a stage-2
fine-tune, build a dir whose non-transformer components symlink the base and whose transformer is the
stage-1 output.

  python3 assemble_lingbot_descendant.py \
    --base /workspace/vla/models/lingbot-va-posttrain-robotwin \
    --transformer /workspace/vla/lingbot_out/robotwin_setB_lora/checkpoints/checkpoint_step_2000/transformer \
    --out /workspace/vla/models/lingbot-descendant-robotwin-setB
"""
from __future__ import annotations

import argparse
import os
import pathlib


def assemble(base: pathlib.Path, transformer: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # symlink every base entry except the transformer
    for entry in sorted(base.iterdir()):
        if entry.name == "transformer":
            continue
        link = out / entry.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(entry.resolve())
    # point transformer at the trained one
    tlink = out / "transformer"
    if tlink.exists() or tlink.is_symlink():
        tlink.unlink()
    tlink.symlink_to(transformer.resolve())
    print(f"assembled {out}:")
    for e in sorted(out.iterdir()):
        tgt = os.readlink(e) if e.is_symlink() else "(dir)"
        print(f"  {e.name} -> {tgt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--transformer", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    if not a.transformer.exists():
        raise SystemExit(f"transformer not found: {a.transformer} (train stage-1 first)")
    assemble(a.base, a.transformer, a.out)


if __name__ == "__main__":
    main()
