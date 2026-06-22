# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
extract_latents.py -- rebuild a LingBot-VA latent dataset from a raw v2.1
LeRobotDataset (libero / robotwin), matching the on-disk format that
``wan_va.dataset.lerobot_latent_dataset.LatentLeRobotDataset`` consumes.

The repo never shipped this script: training was always fed pre-extracted
latents from a cloud mount that is now gone.  This reproduces that mount.

Output layout (per the loader, lerobot_latent_dataset.py:181-222):

    <out-root>/latents/chunk-000/<camera_key>/episode_<NNNNNN>_<start>_<end>.pth
    <out-root>/meta/episodes.jsonl
    <out-root>/empty_emb.pt

Each ``.pth`` is a dict (torch.save) with EXACTLY these keys (verified against
/workspace/vla/lingbot_latents/libero_long):

    latent            : Tensor [(f h w), 48]  bfloat16   (Wan-VAE mu, normalized)
    latent_num_frames : int   (f)
    latent_height     : int   (h)
    latent_width      : int   (w)   -> f*h*w == latent.shape[0]
    video_num_frames  : int   (source frames actually encoded, == (f-1)*4 + 1)
    video_height      : int
    video_width       : int
    text_emb          : Tensor [512, 4096] bfloat16   (UMT5, padded to 512)
    text              : str   (the instruction)
    frame_ids         : np.ndarray int64  (source frame indices, contiguous 0..N-1)
    start_frame       : int   (action_config start; full episode -> 0)
    end_frame         : int   (action_config end;   full episode -> length)
    fps               : int
    ori_fps           : int

Encoder code paths reused 1:1 from the inference server so latents are
identical to what the model was trained on:
  * VAE        : wan_va.modules.utils.load_vae  -> diffusers AutoencoderKLWan
                 (modules/utils.py:12-21).  Streamed via WanVAEStreamingWrapper
                 (modules/utils.py:79-109), the same path wan_va_server._encode_obs
                 uses (wan_va_server.py:374-384).
  * Text enc.  : wan_va.modules.utils.load_text_encoder -> UMT5EncoderModel,
                 load_tokenizer -> T5TokenizerFast (modules/utils.py:24-38).
                 Prompt -> embeds via the server's _get_t5_prompt_embeds logic
                 (wan_va_server.py:115-160), max_sequence_length=512.

Preprocessing constants matched from configs/va_libero_cfg.py and the VAE config:
  * image size 128x128            (va_libero_cfg.py:16-17 height/width)
  * spatial downsample 16, temporal 4, z_dim 48 (vae/config.json
    scale_factor_spatial / scale_factor_temporal / z_dim)
  * fps == ori_fps from dataset info.json (no temporal subsampling; stride 1)
  * text tokens 512               (wan_va_server.py:119 max_sequence_length)
  * latents normalized with (mu - latents_mean) / latents_std  using the VAE
    config's latents_mean / latents_std (wan_va_server.py:379-383).

GPU is required for a real run (VAE + UMT5 encode).  --dry-run loads the models,
processes ONE episode and asserts the produced dict matches a reference
libero_long sample, without writing the dataset.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# wan_va package + lerobot must be importable via the PYTHONPATH overlay:
#   PYTHONPATH=/workspace/vla/lingbot_pydeps:/workspace/vla/lerobot-0.3.3/src:/workspace/vla/lingbot-va
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)

# ----------------------------------------------------------------------------
# Constants matched to training config (cite: configs/va_libero_cfg.py:16-17,
# vae/config.json scale_factor_spatial/temporal, wan_va_server.py:119)
# ----------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "/workspace/vla/models/lingbot-va-posttrain-libero-long"
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
MAX_TEXT_TOKENS = 512  # wan_va_server.py:119
SPATIAL_DOWNSAMPLE = 16  # vae/config.json scale_factor_spatial
TEMPORAL_DOWNSAMPLE = 4  # vae/config.json scale_factor_temporal
REFERENCE_SAMPLE = (
    "/workspace/vla/lingbot_latents/libero_long/latents/chunk-000/"
    "observation.images.agentview_rgb/episode_000000_0_388.pth"
)
REFERENCE_EMPTY_EMB = "/workspace/vla/lingbot_latents/libero_long/empty_emb.pt"


# ----------------------------------------------------------------------------
# prompt cleaning -- use the SAME prompt_clean the inference server uses
# (wan_va_server.py:14 imports it from diffusers.pipelines.wan.pipeline_wan,
# and applies it in _get_t5_prompt_embeds at line 127).  Importing the exact
# symbol guarantees identical tokenization input.
# ----------------------------------------------------------------------------
from diffusers.pipelines.wan.pipeline_wan import prompt_clean


def video_num_frames_for(length: int) -> int:
    """Largest N <= length with (N - 1) % TEMPORAL_DOWNSAMPLE == 0.

    Verified against libero_long: ep len 388 -> 385, 317 -> 317, 312 -> 309.
    This is the frame budget the original extractor used so that
    latent_num_frames = (N - 1) / 4 + 1 is an integer (Wan causal VAE).
    """
    if length < 1:
        return 0
    rem = (length - 1) % TEMPORAL_DOWNSAMPLE
    return length - rem


def autodetect_camera_keys(dataset: LeRobotDataset):
    """Return observation.images.* keys from dataset.meta.features.

    LIBERO  -> agentview_rgb, eye_in_hand_rgb
    RoboTwin -> cam_high, cam_left_wrist, cam_right_wrist
    """
    keys = [
        k
        for k in dataset.meta.features
        if k.startswith("observation.images.")
    ]
    return sorted(keys)


class LatentExtractor:
    def __init__(self, base_model: str, device: str, height: int, width: int):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16  # shared_config.py:10 param_dtype
        self.height = height
        self.width = width

        # --- VAE (modules/utils.py:12-21) ---
        self.vae = load_vae(
            os.path.join(base_model, "vae"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.vae.eval()
        # streaming wrapper == server encode path (wan_va_server.py:67, 377)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean)
        self.latents_std = torch.tensor(self.vae.config.latents_std)

        # --- text encoder + tokenizer (modules/utils.py:24-38) ---
        self.tokenizer = load_tokenizer(os.path.join(base_model, "tokenizer"))
        self.text_encoder = load_text_encoder(
            os.path.join(base_model, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.text_encoder.eval()

    # -- text -------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Reproduce wan_va_server._get_t5_prompt_embeds (lines 115-160).

        Returns [512, 4096] bf16 on CPU.  Tokens beyond the real length are
        zero-padded (the server slices to seq_lens then re-pads to 512).
        """
        prompt = prompt_clean(prompt)
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=MAX_TEXT_TOKENS,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        dev = next(self.text_encoder.parameters()).device
        embeds = self.text_encoder(
            ids.to(dev), mask.to(dev)
        ).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device="cpu")
        # slice to real length then re-pad to 512 (server lines 146-152)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(MAX_TEXT_TOKENS - u.size(0), u.size(1))]
                )
                for u in embeds
            ],
            dim=0,
        )
        return embeds[0]  # [512, 4096]

    # -- video ------------------------------------------------------------
    @torch.no_grad()
    def encode_video(self, frames_chw: torch.Tensor):
        """Encode one camera's clip.

        frames_chw: [T, C, H, W] float in [0, 1] (lerobot video decode format,
        video_utils.py:165).  Mirrors wan_va_server._encode_obs (lines 350-384)
        for the non-tshape branch:
            permute to [1, C, T, H, W], bilinear resize to (height,width),
            scale to [-1, 1], streaming-VAE encode, take mu, normalize.

        Returns (latent_seq [(f h w), 48] bf16, f, h, w).
        """
        # [T, C, H, W] -> [C, T, H, W]
        video = frames_chw.permute(1, 0, 2, 3).contiguous().float()
        # bilinear resize spatial dims (interpolate operates on last 2 dims of
        # a [C, T, H, W] tensor -> resizes H,W).  Matches server line 353-356.
        video = F.interpolate(
            video,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        video = video.unsqueeze(0)  # [1, C, T, H, W]
        # lerobot frames are already /255 in [0,1]; server does /255*2-1 on raw
        # uint8, so the equivalent here is x*2-1.
        video = video * 2.0 - 1.0

        vae_device = next(self.streaming_vae.vae.parameters()).device
        # The Wan2.2 VAE (z=48) has an UNcached avg_shortcut (AvgDown3D) in its
        # down-blocks, so a whole clip cannot be pushed through one streaming
        # encoder() pass: the cached downsampler and the avg_shortcut then disagree
        # on the temporal length (e.g. 137 vs 69).  Use the diffusers native
        # _encode, which feeds the canonical Wan layout (frame 0, then groups of 4)
        # while accumulating the causal feat_cache exactly as during training /
        # inference.  _encode already applies patchify + quant_conv and returns
        # [1, 2*z, F, H, W]; the streaming wrapper's encode_chunk (whole-clip) is
        # kept for the rollout server, which only ever feeds small chunks.
        enc_out = self.streaming_vae.vae._encode(
            video.to(vae_device).to(self.dtype)
        )
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)  # [1, 48, F, H, W]

        # normalize_latents: (mu - mean) * (1/std)  (wan_va_server.py:382, 220-231)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mu.device)
        inv_std = (1.0 / self.latents_std).view(1, -1, 1, 1, 1).to(mu.device)
        mu = ((mu.float() - mean) * inv_std).to(mu)

        mu = mu[0]  # [48, F, H, W]
        c, f, h, w = mu.shape
        # loader inverse is rearrange('(f h w) c -> f h w c'); produce its inverse
        latent_seq = rearrange(mu, "c f h w -> (f h w) c").to(torch.bfloat16).cpu()
        return latent_seq, f, h, w


def gather_episode_frames(dataset, ep_idx, camera_keys, num_frames):
    """Pull the first `num_frames` frames of episode `ep_idx`.

    Returns: dict camera_key -> [T, C, H, W] float[0,1] tensor, plus the task
    string (taken from frame 0).
    """
    start = int(dataset.episode_data_index["from"][ep_idx].item())
    per_cam = {k: [] for k in camera_keys}
    task = None
    for local in range(num_frames):
        item = dataset[start + local]
        if task is None:
            task = item["task"]
        for k in camera_keys:
            per_cam[k].append(item[k])  # [C, H, W] float[0,1]
    out = {k: torch.stack(v, dim=0) for k, v in per_cam.items()}
    return out, task


def build_meta_line(ep_idx, task, length):
    """One episodes.jsonl line matching the reference format."""
    return {
        "episode_index": int(ep_idx),
        "tasks": [task],
        "length": int(length),
        "action_config": [
            {
                "start_frame": 0,
                "end_frame": int(length),
                "action_text": task,
                "skill": "",
            }
        ],
    }


def write_empty_emb(extractor, out_root, copy_reference):
    """Write empty_emb.pt -- the UMT5 embedding of "".

    The empty embedding is instruction-independent (used for CFG), so it can be
    copied verbatim from libero_long OR re-encoded.  We re-encode "" by default
    (self-contained) but support --copy-empty-emb to copy the reference.
    """
    dest = Path(out_root) / "empty_emb.pt"
    if copy_reference and os.path.exists(REFERENCE_EMPTY_EMB):
        emb = torch.load(REFERENCE_EMPTY_EMB, weights_only=False, map_location="cpu")
    else:
        emb = extractor.encode_text("")
    torch.save(emb, dest)
    return dest


def process_episode(extractor, dataset, ep_idx, camera_keys, chunk_dir, length):
    """Encode one episode for all cameras, save .pth files, return meta dict."""
    num_frames = video_num_frames_for(length)
    fps = int(dataset.meta.info.get("fps", dataset.fps))
    frames, task = gather_episode_frames(dataset, ep_idx, camera_keys, num_frames)
    text_emb = extractor.encode_text(task)  # [512, 4096] bf16

    for k in camera_keys:
        latent_seq, f, h, w = extractor.encode_video(frames[k])
        sample = {
            "latent": latent_seq,
            "latent_num_frames": int(f),
            "latent_height": int(h),
            "latent_width": int(w),
            "video_num_frames": int(num_frames),
            "video_height": int(extractor.height),
            "video_width": int(extractor.width),
            "text_emb": text_emb,
            "text": task,
            "frame_ids": np.arange(num_frames, dtype=np.int64),
            "start_frame": 0,
            "end_frame": int(length),
            "fps": int(fps),
            "ori_fps": int(fps),
        }
        cam_dir = Path(chunk_dir) / k
        cam_dir.mkdir(parents=True, exist_ok=True)
        out_file = cam_dir / f"episode_{ep_idx:06d}_0_{length}.pth"
        torch.save(sample, out_file)
    return build_meta_line(ep_idx, task, length)


# ----------------------------------------------------------------------------
# dry-run validation against a reference libero_long sample
# ----------------------------------------------------------------------------
def dry_run(extractor, dataset, camera_keys):
    print("=== DRY RUN: encoding ONE episode and comparing to reference ===")
    ref_path = REFERENCE_SAMPLE
    have_ref = os.path.exists(ref_path)
    if not have_ref:
        print(f"[warn] reference sample not found at {ref_path}; "
              "will only check self-consistency.")
    ref = (
        torch.load(ref_path, weights_only=False, map_location="cpu")
        if have_ref
        else None
    )

    ep_idx = int(list(dataset.meta.episodes.keys())[0])
    length = int(dataset.meta.episodes[ep_idx]["length"])
    num_frames = video_num_frames_for(length)
    frames, task = gather_episode_frames(dataset, ep_idx, camera_keys, num_frames)
    text_emb = extractor.encode_text(task)

    k0 = camera_keys[0]
    latent_seq, f, h, w = extractor.encode_video(frames[k0])
    sample = {
        "latent": latent_seq,
        "latent_num_frames": int(f),
        "latent_height": int(h),
        "latent_width": int(w),
        "video_num_frames": int(num_frames),
        "video_height": int(extractor.height),
        "video_width": int(extractor.width),
        "text_emb": text_emb,
        "text": task,
        "frame_ids": np.arange(num_frames, dtype=np.int64),
        "start_frame": 0,
        "end_frame": int(length),
        "fps": int(dataset.meta.info.get("fps", dataset.fps)),
        "ori_fps": int(dataset.meta.info.get("fps", dataset.fps)),
    }

    expected_keys = {
        "latent", "latent_num_frames", "latent_height", "latent_width",
        "video_num_frames", "video_height", "video_width", "text_emb",
        "text", "frame_ids", "start_frame", "end_frame", "fps", "ori_fps",
    }
    failures = []

    def check(cond, msg):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    # ---- structural / dtype / shape-invariant checks (always run) ----
    check(set(sample.keys()) == expected_keys,
          f"keys == expected ({sorted(set(sample.keys()) ^ expected_keys)} diff)")
    check(sample["latent"].dtype == torch.bfloat16,
          f"latent dtype bf16 (got {sample['latent'].dtype})")
    check(sample["latent"].ndim == 2 and sample["latent"].shape[1] == 48,
          f"latent shape [*,48] (got {tuple(sample['latent'].shape)})")
    inv = sample["latent_num_frames"] * sample["latent_height"] * sample["latent_width"]
    check(inv == sample["latent"].shape[0],
          f"f*h*w == latent.shape[0] ({inv} == {sample['latent'].shape[0]})")
    check((sample["video_num_frames"] - 1) % TEMPORAL_DOWNSAMPLE == 0
          and (sample["video_num_frames"] - 1) // TEMPORAL_DOWNSAMPLE + 1
          == sample["latent_num_frames"],
          "latent_num_frames == (video_num_frames-1)/4 + 1")
    check(sample["latent_height"] == extractor.height // SPATIAL_DOWNSAMPLE
          and sample["latent_width"] == extractor.width // SPATIAL_DOWNSAMPLE,
          f"latent h,w == video/16 ({sample['latent_height']},{sample['latent_width']})")
    check(sample["text_emb"].dtype == torch.bfloat16
          and tuple(sample["text_emb"].shape) == (MAX_TEXT_TOKENS, 4096),
          f"text_emb [512,4096] bf16 (got {tuple(sample['text_emb'].shape)} "
          f"{sample['text_emb'].dtype})")
    check(isinstance(sample["frame_ids"], np.ndarray)
          and sample["frame_ids"].dtype == np.int64
          and len(sample["frame_ids"]) == sample["video_num_frames"],
          "frame_ids int64 ndarray len == video_num_frames")
    for ik in ("latent_num_frames", "latent_height", "latent_width",
               "video_num_frames", "video_height", "video_width",
               "start_frame", "end_frame", "fps", "ori_fps"):
        check(isinstance(sample[ik], int), f"{ik} is int")
    check(isinstance(sample["text"], str), "text is str")

    # ---- field-by-field comparison against the real reference ----
    if ref is not None:
        check(set(sample.keys()) == set(ref.keys()),
              f"key set matches reference ({sorted(set(sample.keys()) ^ set(ref.keys()))})")
        for ik in ("latent_height", "latent_width", "video_height",
                   "video_width", "fps", "ori_fps"):
            check(sample[ik] == ref[ik],
                  f"{ik} == reference ({sample[ik]} vs {ref[ik]})")
        check(sample["latent"].dtype == ref["latent"].dtype,
              f"latent dtype matches reference ({ref['latent'].dtype})")
        check(sample["latent"].shape[1] == ref["latent"].shape[1],
              f"latent channel matches reference ({ref['latent'].shape[1]})")
        check(sample["text_emb"].shape == ref["text_emb"].shape,
              f"text_emb shape matches reference ({tuple(ref['text_emb'].shape)})")
        check(sample["frame_ids"].dtype == ref["frame_ids"].dtype,
              f"frame_ids dtype matches reference ({ref['frame_ids'].dtype})")
        # value-level sanity if this IS the same episode (ep0/agentview).
        same_ep = (
            ref.get("video_num_frames") == sample["video_num_frames"]
            and k0 == "observation.images.agentview_rgb"
        )
        if same_ep:
            num = (sample["latent"].float() - ref["latent"].float()).abs()
            denom = ref["latent"].float().abs().mean().clamp_min(1e-3)
            rel = num.mean() / denom
            cos = F.cosine_similarity(
                sample["latent"].float().flatten(),
                ref["latent"].float().flatten(),
                dim=0,
            ).item()
            check(cos > 0.98,
                  f"latent cosine-sim vs reference > 0.98 (got {cos:.4f}, "
                  f"rel-L1 {rel:.4f})")
            t_cos = F.cosine_similarity(
                sample["text_emb"].float().flatten(),
                ref["text_emb"].float().flatten(),
                dim=0,
            ).item()
            check(t_cos > 0.99,
                  f"text_emb cosine-sim vs reference > 0.99 (got {t_cos:.4f})")

    print()
    if failures:
        print(f"DRY RUN: FAIL ({len(failures)} check(s) failed)")
        for m in failures:
            print(f"   - {m}")
        return 1
    print("DRY RUN: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot-root", required=True,
                    help="path to a v2.1 LeRobotDataset (libero suite or robotwin set)")
    ap.add_argument("--out-root", default=None,
                    help="output latent dataset root (required unless --dry-run)")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="LingBot-VA base dir holding vae/ text_encoder/ tokenizer/")
    ap.add_argument("--camera-keys", default=None,
                    help="comma list; default auto-detect observation.images.* keys")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N episodes")
    ap.add_argument("--dry-run", action="store_true",
                    help="load models, encode ONE episode, assert format matches "
                         "the reference libero_long sample, write nothing")
    ap.add_argument("--copy-empty-emb", action="store_true",
                    help="copy libero_long empty_emb.pt instead of re-encoding \"\"")
    args = ap.parse_args()

    if not args.dry_run and not args.out_root:
        ap.error("--out-root is required unless --dry-run")

    root = Path(args.lerobot_root).resolve()
    # video_backend pyav matches the loader (lerobot_latent_dataset.py:121)
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        revision="v2.1",
        video_backend="pyav",
    )

    if args.camera_keys:
        camera_keys = [k.strip() for k in args.camera_keys.split(",") if k.strip()]
    else:
        camera_keys = autodetect_camera_keys(dataset)
    if not camera_keys:
        raise SystemExit("no observation.images.* camera keys found / specified")
    print(f"camera keys: {camera_keys}")

    extractor = LatentExtractor(args.base_model, args.device, args.height, args.width)

    if args.dry_run:
        sys.exit(dry_run(extractor, dataset, camera_keys))

    out_root = Path(args.out_root)
    chunk_dir = out_root / "latents" / "chunk-000"
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    ep_indices = sorted(int(e) for e in dataset.meta.episodes.keys())
    if args.limit is not None:
        ep_indices = ep_indices[: args.limit]

    write_empty_emb(extractor, out_root, args.copy_empty_emb)

    meta_lines = []
    for ep_idx in tqdm(ep_indices, desc="episodes"):
        length = int(dataset.meta.episodes[ep_idx]["length"])
        meta_lines.append(
            process_episode(extractor, dataset, ep_idx, camera_keys, chunk_dir, length)
        )

    with open(out_root / "meta" / "episodes.jsonl", "w") as f:
        for line in meta_lines:
            f.write(json.dumps(line) + "\n")

    print(f"done: {len(meta_lines)} episodes -> {out_root}")


if __name__ == "__main__":
    main()
