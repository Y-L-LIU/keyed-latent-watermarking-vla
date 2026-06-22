"""STAGE 2/4: train one LoRA student by BC on a relabeled corpus.

Thin wrapper over wan_va.train.Trainer that overrides dataset_path / save_root / step
budget / save cadence from env, so the shared repo config is untouched. Saves ONLY the
final merged transformer (save_interval == num_steps) to keep disk lean (~one ckpt).

Env:
  DATASET_PATH  relabeled dataset dir (latents symlinked)
  SAVE_ROOT     where checkpoints/ go
  NUM_STEPS     training steps (default 1500)
  BASE_CKPT     base teacher/init ckpt
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, "/workspace/vla/lingbot-va")
sys.path.insert(0, "/workspace/vla/lingbot-va/wan_va")


def main():
    from configs import VA_CONFIGS
    from distributed.util import init_distributed
    from train import Trainer

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    cfg = VA_CONFIGS["libero_lora_train"]
    cfg.rank = rank; cfg.local_rank = local_rank; cfg.world_size = world_size

    cfg.wan22_pretrained_model_name_or_path = os.environ.get(
        "BASE_CKPT", "/workspace/vla/models/lingbot-va-posttrain-libero-long")
    cfg.dataset_path = os.environ["DATASET_PATH"]
    cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
    cfg.save_root = os.environ["SAVE_ROOT"]
    cfg.num_steps = int(os.environ.get("NUM_STEPS", 1500))
    # save ONLY the final checkpoint (disk is tight)
    cfg.save_interval = cfg.num_steps
    cfg.enable_wandb = False

    if rank == 0:
        print(f"[train] dataset={cfg.dataset_path}")
        print(f"[train] save_root={cfg.save_root} num_steps={cfg.num_steps} "
              f"save_interval={cfg.save_interval}")
        print(f"[train] base={cfg.wan22_pretrained_model_name_or_path}")

    trainer = Trainer(cfg)
    trainer.train()
    trainer.save_checkpoint()  # ensure a final save even if num_steps%interval timing differs
    if rank == 0:
        print("[train] STUDENT_TRAIN_COMPLETE")


if __name__ == "__main__":
    main()
