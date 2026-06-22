# Copyright 2024-2025.
"""Attack-D (direct adversarial) TRACTION PROBE on LingBot-VA / LIBERO.

Question this answers: does an adaptive adversary that fine-tunes the policy (LoRA)
to defeat the latent-noise watermark detector actually have a gradient handle on
LingBot, the way it did NOT on pi0.5?

  * pi0.5 Attack-D was a clean no-op: the per-episode recovery signal is ~chance
    by construction (under-determined seed), so a per-episode gradient attacker had
    nothing to grip. (see results/score_attackd.py + memory vla-attackd-direct-adversarial)
  * LingBot's per-episode recovery cosine is ~0.4 (10x pi0.5) — a STRONG per-episode
    signal. So the same attack might actually work here. This probe finds out, cheaply,
    WITHOUT env rollout: it is an UPPER BOUND on attack efficacy (penalty-only, the
    attacker does not even pay a task-utility cost). If recovery does not move here,
    the watermark is robust a fortiori. If it does move, escalate to a faithful
    cost-utility build (real FM task loss + env rollout + canonical raw scoring).

Bilevel structure (identical in spirit to finetune_attack_d_jax.py):
  inner: z_hat = unrolled-Adam MAP recovering the action seed from observed actions,
         decoded through the FROZEN BASE policy (LoRA scaling = 0).
  outer: minimize  lambda * (mean active-dim recovery cosine(z_hat, r))^2
         over LoRA params, where r is a random keyed reference injected at beta=1
         into the SUSPECT policy's (LoRA on) action sampler.
The attacker does NOT know the deployed key: each step samples a random key, so the
learned behavior is generic seed-recovery resistance, not key-specific overfitting.

Single GPU. Env (osmesa) used ONCE at startup to render an init-obs bank for cache
priming; the bilevel itself is env-free.

Usage (nodeA):
  cd /workspace/vla/lingbot-va
  export PYTHONPATH=/workspace/vla/lingbot-va:/workspace/vla/openpi/third_party/libero:$PYTHONPATH
  export MUJOCO_GL=osmesa WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=29560 \
    -m wan_va.wm.attack_d_probe_lingbot --lambda-attack 10 --num-steps 300 \
    --bank-tasks 8 --inner-iters 4 --roll-steps 6 --map-steps 6 \
    --out-dir /workspace/vla_out/attack_c/attacked/lingbot_libero_attackd/probe_lam10
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ----------------------------- reference / scoring helpers -----------------------------

def keyed_reference_active(secret_key, episode_nonce, chunk_index, *, length, n_active,
                           control_freq):
    """[n_active, length] keyed gaussian reference for the active action dims (numpy)."""
    from wan_va.wm.watermark import (
        InternalNoiseWatermarkConfig, WatermarkContext, generate_keyed_reference,
    )
    cfg = InternalNoiseWatermarkConfig(secret_key=int(secret_key), control_freq=float(control_freq),
                                       beta=1.0)
    ctx = WatermarkContext(chunk_index=int(chunk_index), episode_nonce=int(episode_nonce))
    ref = generate_keyed_reference(length=length, action_dim=n_active, sample_rate_hz=control_freq,
                                   config=cfg, context=ctx)  # [length, n_active]
    return ref.T.astype(np.float32)  # [n_active, length]


def recovery_cosine_torch(z_active, ref_active):
    """Mean per-dim cosine between recovered active noise and reference (differentiable).

    z_active, ref_active: [n_active, length] tensors. Mirrors scoring.build_score_vector_from_noise
    (mean-subtract each dim, normalized dot), then averages over dims -> scalar matched filter.
    """
    z = z_active - z_active.mean(dim=1, keepdim=True)
    r = ref_active - ref_active.mean(dim=1, keepdim=True)
    zn = z.norm(dim=1)
    rn = r.norm(dim=1)
    denom = (zn * rn).clamp_min(1e-8)
    cos = (z * r).sum(dim=1) / denom
    return cos.mean()


# ----------------------------- LoRA on/off context -----------------------------

class lora_disabled:
    """Temporarily zero every LoRALinear.lora_scaling buffer -> frozen BASE policy.

    Used for the verifier's MAP inversion (detector loads the base, not the suspect).
    Restores the original scalings on exit. No grad to LoRA flows through a disabled
    forward (scaling buffer is 0), exactly matching the threat model where the verifier
    runs the public base checkpoint.
    """

    def __init__(self, transformer):
        self.t = transformer
        self.saved = []

    def __enter__(self):
        from wan_va.lora import LoRALinear
        for m in self.t.modules():
            if isinstance(m, LoRALinear):
                self.saved.append((m, m.lora_scaling.clone()))
                m.lora_scaling.zero_()
        return self

    def __exit__(self, *a):
        for m, val in self.saved:
            m.lora_scaling.copy_(val)
        self.saved = []
        return False


# ----------------------------- unrolled differentiable MAP -----------------------------

def unrolled_map(decode_fn, obs_op, y_obs, z_shape, *, n_iters, lr, obs_sigma, prior_weight,
                 device, dtype, clip=10.0):
    """Differentiable MAP: a few gradient-descent steps recovering z from y_obs.

    Graph is retained (create_graph) so d(z_hat)/d(y_obs) flows (2nd-order); grad to LoRA
    enters via y_obs and via decode_fn (suspect self-inversion). PLAIN GD, not Adam: Adam's
    mhat/(sqrt(vhat)+eps) makes the double-backward graph blow up when vhat->0 (-> NaN). With
    obs_sigma=1 the residual term is O(1), so GD with a moderate lr is well-scaled and the
    second-order graph stays finite. Per-step grad-norm clip + non-finite zeroing as a backstop.
    """
    z = torch.zeros(*z_shape, device=device, dtype=dtype).requires_grad_(True)  # mean-of-prior start
    inv_sig2 = 1.0 / (obs_sigma * obs_sigma)
    for t in range(n_iters):
        a_pred = decode_fn(z)
        pred_obs = obs_op.apply(a_pred)
        obs_loss = 0.5 * inv_sig2 * ((pred_obs - y_obs) ** 2).mean()
        prior_loss = 0.5 * prior_weight * z.square().mean()
        loss = obs_loss + prior_loss
        g = torch.autograd.grad(loss, z, create_graph=True)[0]
        gn = g.norm()
        scale = torch.where(gn > clip, clip / (gn + 1e-12), torch.ones_like(gn))
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * scale
        z = z - lr * g
    return z


# ----------------------------- obs bank (env, once) -----------------------------

def build_obs_bank(suite, n_tasks, seed=0):
    """Render one initial observation + prompt per task (proven eval priming path)."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from wan_va.wm.eval_libero_watermark import _extract_obs

    bdict = benchmark.get_benchmark_dict()
    binst = bdict[suite]()
    ntotal = binst.get_num_tasks()
    n_tasks = min(n_tasks, ntotal)
    bank = []
    for ti in range(n_tasks):
        task = binst.get_task(ti)
        init_states = binst.get_task_init_states(ti)
        env_args = {"bddl_file_name": binst.get_task_bddl_file_path(ti),
                    "camera_heights": 128, "camera_widths": 128}
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        env.set_init_state(init_states[0])
        obs_raw = None
        for _ in range(5):
            obs_raw, _, _, _ = env.step([0.0] * 7)
        bank.append({"prompt": task.language, "obs": _extract_obs(obs_raw)})
        env.close()
        print(f"[obs-bank] task {ti}: {task.language}", flush=True)
    return bank


def build_obs_bank_robotwin(obs_bank_file, n_tasks):
    """Load a pre-rendered RoboTwin initial-observation bank.

    The bank is written by dump_robotwin_obs_bank.py as object arrays because each entry
    is the exact multi-camera/state dict consumed by VA_Server._infer.
    """
    p = Path(obs_bank_file)
    if not p.exists():
        raise FileNotFoundError(
            f"RoboTwin obs bank not found: {p}. Run "
            "`python -m wan_va.wm.dump_robotwin_obs_bank --robotwin-root /workspace/vla/RoboTwin "
            f"--out {p}` first."
        )
    d = np.load(p, allow_pickle=True)
    prompts = [str(x) for x in d["prompts"].tolist()]
    obs = d["obs"].tolist()
    n = min(int(n_tasks), len(prompts))
    bank = [{"prompt": prompts[i], "obs": obs[i]} for i in range(n)]
    for i, entry in enumerate(bank):
        task = str(d["task_names"][i]) if "task_names" in d.files else f"task{i}"
        print(f"[obs-bank-rt] {i}: {task} | {entry['prompt']}", flush=True)
    return bank


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", type=str, default="libero_descendant",
                    help="base policy config (transformer = base posttrain, NOT merged descendant)")
    ap.add_argument("--base-config-name", type=str, default="libero",
                    help="config whose transformer is the clean base posttrain checkpoint")
    ap.add_argument("--domain", type=str, choices=["libero", "robotwin"], default="libero",
                    help="observation-bank source and dynamic defaults")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--obs-bank-file", type=str,
                    default="/workspace/vla_out/attack_c/obs_banks/lingbot_robotwin_init_obs_bank.npz",
                    help="pre-rendered RoboTwin initial observations (used with --domain robotwin)")
    ap.add_argument("--bank-tasks", type=int, default=8)
    ap.add_argument("--lambda-attack", type=float, default=10.0)
    ap.add_argument("--task-weight", type=float, default=0.0,
                    help="weight on the clean-behavior preservation loss: keep the suspect's "
                         "deployed (watermarked) actions close to the BASE policy's actions on "
                         "actuated dims. 0 = penalty-only probe (attack upper bound).")
    ap.add_argument("--num-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-l2", type=float, default=1e-4)
    ap.add_argument("--save-final", action="store_true",
                    help="merge LoRA and write a loadable model dir (base symlinks + merged transformer)")
    ap.add_argument("--base-model-dir", type=str,
                    default="/workspace/vla/models/lingbot-va-posttrain-libero-long",
                    help="base dir whose non-transformer components are symlinked into the saved suspect")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--roll-steps", type=int, default=6, help="flow steps for suspect forward roll")
    ap.add_argument("--map-steps", type=int, default=6, help="flow steps inside MAP decode")
    ap.add_argument("--inner-iters", type=int, default=4, help="unrolled MAP Adam iters")
    ap.add_argument("--map-lr", type=float, default=0.08)
    ap.add_argument("--obs-sigma", type=float, default=1e-3)
    ap.add_argument("--map-prior-weight", type=float, default=1.0)
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="single-process model parallelism via device_map=balanced")
    ap.add_argument("--parallel-mode", type=str, choices=["auto", "mp", "fsdp"], default="auto",
                    help="auto/mp: single-process device_map model parallelism; "
                         "fsdp: torchrun multi-process FSDP data parallelism with LoRA "
                         "injected before sharding")
    ap.add_argument("--max-memory", type=str, nargs="+", default=None,
                    help="optional per-visible-GPU memory caps for accelerate device_map")
    # inner unrolled-MAP (bilevel) knobs, decoupled from the eval solver: obs_sigma=1 keeps
    # the residual O(1) so plain-GD double-backward stays finite.
    ap.add_argument("--inner-lr", type=float, default=0.5)
    ap.add_argument("--inner-obs-sigma", type=float, default=1.0)
    ap.add_argument("--secret-key-base", type=int, default=42)
    ap.add_argument("--eval-interval", type=int, default=25)
    ap.add_argument("--eval-keys", type=int, default=8)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.domain == "robotwin":
        if args.base_config_name == "libero":
            args.base_config_name = "robotwin"
        if args.config_name == "libero_descendant":
            args.config_name = "robotwin_descendant"
        if args.base_model_dir == "/workspace/vla/models/lingbot-va-posttrain-libero-long":
            args.base_model_dir = "/workspace/vla/models/lingbot-va-posttrain-robotwin"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    from wan_va.distributed.util import init_distributed
    init_distributed(world_size, local_rank, rank)
    parallel_mode = args.parallel_mode
    if parallel_mode == "auto":
        parallel_mode = "fsdp" if world_size > 1 else "mp"
    if world_size > 1 and parallel_mode != "fsdp":
        raise ValueError("multi-process Attack-D requires --parallel-mode fsdp")
    if parallel_mode == "fsdp" and args.num_gpus > 1:
        raise ValueError("--parallel-mode fsdp uses one visible GPU per rank; do not combine it with --num-gpus > 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # The unrolled MAP needs double-backward (create_graph=True). Flash/mem-efficient SDPA
    # kernels have no backward-of-backward; force the math kernel which supports higher-order
    # grad. Action-token attention is tiny (16 tokens) so the cost is negligible; video priming
    # runs under no_grad regardless.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from wan_va.wan_va_server import VA_Server
    from wan_va.configs import VA_CONFIGS
    from wan_va.lora import inject_lora, mark_only_lora_trainable
    from wan_va.wm.observation import ChannelObservation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load BASE posttrain policy as the server (suspect = base + trainable LoRA) ---
    cfg = VA_CONFIGS[args.base_config_name]
    cfg.rank = rank
    cfg.local_rank = local_rank
    cfg.world_size = world_size
    cfg.enable_offload = True
    if parallel_mode == "fsdp":
        cfg.pre_shard_lora_rank = args.lora_rank
        cfg.pre_shard_lora_alpha = args.lora_alpha
        cfg.pre_shard_lora_dropout = 0.0
        cfg.pre_shard_lora_include_ffn = False
        cfg.pre_shard_apply_ac = True
    elif args.num_gpus > 1:
        cfg.device_map = "balanced"
    if args.max_memory:
        cfg.max_memory_list = args.max_memory
    cfg.save_root = str(out_dir / "server_scratch")
    (out_dir / "server_scratch").mkdir(parents=True, exist_ok=True)
    # we never use the priming dumps; no-op save_async so 300 priming steps don't spam disk
    import wan_va.wan_va_server as _wvs
    _wvs.save_async = lambda *a, **k: None
    print(f"[load] VA_Server base={cfg.wan22_pretrained_model_name_or_path}", flush=True)
    server = VA_Server(cfg)

    # inject fresh trainable LoRA into the loaded transformer (base frozen)
    if parallel_mode == "fsdp":
        n = int(getattr(server, "_pre_shard_lora_wrapped", 0))
    else:
        n = inject_lora(server.transformer, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0)
    if parallel_mode != "fsdp" and not hasattr(server.transformer, "hf_device_map"):
        server.transformer.to(server.device)
    trainable, total = mark_only_lora_trainable(server.transformer)
    server.transformer.train()  # enable grad paths; base params stay requires_grad=False
    print(f"[lora] wrapped {n} linears, trainable {trainable:,}/{total:,} "
          f"({100*trainable/max(total,1):.4f}%)", flush=True)

    lora_params = [p for p in server.transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    fcs = server.job_config.frame_chunk_size
    apf = server.job_config.action_per_frame
    length = fcs * apf
    active = list(server.job_config.used_action_channel_ids)
    n_active = len(active)
    adim = server.job_config.action_dim
    control_freq = float(length)
    obs_op = ChannelObservation(channel_idx=tuple(active))
    z_shape = (1, adim, fcs, apf, 1)
    dev, dt = server.device, server.dtype
    dist_on = torch.distributed.is_initialized() and world_size > 1

    def dist_avg_scalar(x):
        t = torch.as_tensor(float(x), device=dev, dtype=torch.float32)
        if dist_on:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
        return float(t.detach().cpu())

    def dist_all_finite(ok):
        t = torch.tensor(1 if ok else 0, device=dev, dtype=torch.int32)
        if dist_on:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        return bool(int(t.detach().cpu()))

    if rank == 0:
        print(f"[parallel] mode={parallel_mode} world_size={world_size} local_rank={local_rank} "
              f"num_gpus={args.num_gpus}", flush=True)

    # --- obs bank (env, once) ---
    if args.domain == "robotwin":
        bank = build_obs_bank_robotwin(args.obs_bank_file, args.bank_tasks)
    else:
        bank = build_obs_bank(args.suite, args.bank_tasks, seed=args.seed)
    nb = len(bank)

    from wan_va.wm.watermark import InternalNoiseWatermarkConfig, WatermarkContext

    def prime_and_get_refnoise(entry, secret_key, episode_nonce):
        """Prime KV-cache with one obs at beta=1 for the given key; return injected wm noise.

        Runs the proven _infer path (no_grad) which primes the video cache AND stores the
        watermarked initial noise in server._last_wm_noise (= z_inj, the MAP seed/target seed).
        """
        wm_cfg = InternalNoiseWatermarkConfig(secret_key=int(secret_key), control_freq=control_freq,
                                              beta=1.0, chunk_selection_period=1,
                                              chunk_selection_count=1, chunk_start_min=0)
        ctx = WatermarkContext(chunk_index=0, episode_nonce=int(episode_nonce))
        server._reset(prompt=entry["prompt"])
        with torch.no_grad():
            server._infer({"obs": [entry["obs"]]}, frame_st_id=0, wm_config=wm_cfg, wm_context=ctx)
        return server._last_wm_noise.detach().clone()  # [1, adim, fcs, apf, 1]

    def ref_active_tensor(secret_key, episode_nonce):
        ref = keyed_reference_active(secret_key, episode_nonce, 0, length=length,
                                     n_active=n_active, control_freq=control_freq)
        return torch.from_numpy(ref).to(dev).float()  # [n_active, length]

    def decode_active(z_full):
        """[n_active, length] active-channel view of the recovered/decoded action noise."""
        # z_full: [1, adim, fcs, apf, 1] -> active -> [n_active, length]
        return z_full[0, active].reshape(n_active, length)

    # ===================== faithful detector eval (LoRA on/off) =====================
    # NOT under no_grad: the real FMLatentMAPSolver optimizes its own z via autograd.
    # We detach the rolled actions so only the solver's small z-graph is built (no LoRA graph).
    def evaluate(tag, step):
        """Real recovery cosine via the actual FMLatentMAPSolver, suspect (LoRA on) vs base.

        Reports mean over eval keys/tasks of the per-dim recovery cosine of the OWNER key,
        the quantity the matched-filter detector integrates. This is the ground-truth signal.
        """
        from wan_va.wm.fm_latent_map_solver import FMLatentMAPSolver, FMLatentMAPConfig
        map_cfg = FMLatentMAPConfig(num_iters=30, lr=args.map_lr, obs_sigma=args.obs_sigma,
                                    prior_weight=args.map_prior_weight)
        cos_on, cos_base = [], []
        server.transformer.eval()
        for k in range(args.eval_keys):
            sample_k = k * world_size + rank if parallel_mode == "fsdp" else k
            entry = bank[sample_k % nb]
            key = args.secret_key_base
            nonce = 900000 + sample_k
            z_inj = prime_and_get_refnoise(entry, key, nonce)
            ref_a = ref_active_tensor(key, nonce)
            # SUSPECT (attacked): generate AND invert through LoRA-on policy (self-consistent,
            # matching the real lingbot detector which loads a single model for both).
            def dec_suspect(z):
                return server.sample_actions_from_noise(z, frame_st_id=0, num_steps=args.map_steps)
            a_suspect = server.sample_actions_from_noise(z_inj, frame_st_id=0, num_steps=args.roll_steps).detach()
            y_obs = obs_op.apply(a_suspect.float())
            solver = FMLatentMAPSolver(dec_suspect, obs_op, map_cfg)
            zr = solver.solve(y_obs=y_obs, z_init=None, z_shape=z_shape)["z_map"]
            cos_on.append(float(recovery_cosine_torch(decode_active(zr), ref_a)))
            # ANCHOR (lambda=0): generate AND invert through base (LoRA disabled), self-consistent.
            with lora_disabled(server.transformer):
                a_base = server.sample_actions_from_noise(z_inj, frame_st_id=0, num_steps=args.roll_steps).detach()
                y_obs_b = obs_op.apply(a_base.float())
                def dec_base(z):
                    return server.sample_actions_from_noise(z, frame_st_id=0, num_steps=args.map_steps)
                solver_b = FMLatentMAPSolver(dec_base, obs_op, map_cfg)
                zr_b = solver_b.solve(y_obs=y_obs_b, z_init=None, z_shape=z_shape)["z_map"]
                cos_base.append(float(recovery_cosine_torch(decode_active(zr_b), ref_a)))
        server.transformer.train()
        mon = dist_avg_scalar(float(np.mean(cos_on)))
        mbase = dist_avg_scalar(float(np.mean(cos_base)))
        if rank == 0:
            print(f"[eval s{step}] recovery cos  suspect(LoRA on)={mon:+.4f}  "
                  f"base(lam0)={mbase:+.4f}  delta={mon-mbase:+.4f}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"step": step, "cos_suspect": mon, "cos_base": mbase}

    # ===================== attack loop =====================
    log = []
    print(f"[attack] lambda={args.lambda_attack} steps={args.num_steps} "
          f"roll_steps={args.roll_steps} map_steps={args.map_steps} inner={args.inner_iters}", flush=True)
    evaluate("init", 0)
    t0 = time.time()
    for step in range(1, args.num_steps + 1):
        sample_i = step * world_size + rank if parallel_mode == "fsdp" else step
        entry = bank[sample_i % nb]
        if parallel_mode == "fsdp":
            key_rng = np.random.RandomState(args.seed + 1009 * step + 9176 * rank)
            key = args.secret_key_base + int(key_rng.randint(0, 100000))
            nonce = sample_i
        else:
            key = args.secret_key_base + int(np.random.randint(0, 100000))  # attacker does NOT know owner key
            nonce = step

        z_inj = prime_and_get_refnoise(entry, key, nonce)            # cache primed (no_grad)
        ref_a = ref_active_tensor(key, nonce)

        # base target FIRST (task preservation): compute it before the suspect graph exists, so
        # lora_disabled's in-place zero/restore of lora_scaling happens with no live graph
        # referencing the buffer (else backward hits a version-mismatch inplace error).
        y_base = None
        if args.task_weight > 0:
            with torch.no_grad(), lora_disabled(server.transformer):
                a_wm_base = server.sample_actions_from_noise(z_inj, frame_st_id=0,
                                                             num_steps=args.roll_steps)
            y_base = obs_op.apply(a_wm_base.float()).detach()

        # suspect forward roll (LoRA on) -> grad to LoRA
        a_wm = server.sample_actions_from_noise(z_inj, frame_st_id=0, num_steps=args.roll_steps)
        y_obs = obs_op.apply(a_wm.float())

        # inner MAP self-consistently through the SUSPECT (LoRA on), matching the real
        # lingbot detector (single model for gen+invert). Grad to LoRA flows through both
        # the y_obs target AND the decode inside the MAP -> the attacker can break recovery
        # either by changing its actions or by collapsing the seed-sensitivity of inversion.
        def dec_suspect(z):
            return server.sample_actions_from_noise(z, frame_st_id=0, num_steps=args.map_steps)
        z_hat = unrolled_map(dec_suspect, obs_op, y_obs, z_shape, n_iters=args.inner_iters,
                             lr=args.inner_lr, obs_sigma=args.inner_obs_sigma,
                             prior_weight=args.map_prior_weight, device=dev, dtype=torch.float32)

        rec_cos = recovery_cosine_torch(decode_active(z_hat).float(), ref_a)
        l_attack = rec_cos.square()
        l_l2 = torch.zeros((), device=dev)
        for p in lora_params:
            l_l2 = l_l2 + (p.float() ** 2).mean().to(dev)
        l_l2 = l_l2 / max(len(lora_params), 1)

        # task preservation: keep the suspect's DEPLOYED (watermarked) actions close to the BASE
        # policy's actions on the actuated dims. Directly opposes the recovery penalty -> the
        # (task_weight, lambda) frontier is the cost-utility tradeoff. base target is detached.
        l_task = torch.zeros((), device=dev)
        if y_base is not None:
            l_task = ((y_obs - y_base) ** 2).mean()

        loss = args.lambda_attack * l_attack + args.task_weight * l_task + args.lora_l2 * l_l2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # NaN-guard: a single bad 2nd-order step shouldn't poison the LoRA weights.
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in lora_params)
        finite = dist_all_finite(finite)
        if finite:
            gnorm = torch.nn.utils.clip_grad_norm_(lora_params, 2.0)
            opt.step()
        else:
            gnorm = torch.tensor(float("nan"))
            opt.zero_grad(set_to_none=True)

        if step % 5 == 0 or step == 1:
            rec_show = dist_avg_scalar(float(rec_cos.detach()))
            attack_show = dist_avg_scalar(float(l_attack.detach()))
            task_show = dist_avg_scalar(float(l_task.detach()))
            gnorm_show = dist_avg_scalar(float(gnorm.detach())) if torch.is_tensor(gnorm) else dist_avg_scalar(float(gnorm))
            if rank == 0:
                print(f"[s{step:4d}] rec_cos={rec_show:+.4f} l_attack={attack_show:.4f} "
                      f"l_task={task_show:.4f} gnorm={gnorm_show:.2e} key={key} "
                      f"dt={time.time()-t0:.1f}s", flush=True)
                log.append({"step": step, "rec_cos": rec_show, "l_attack": attack_show,
                            "l_task": task_show, "gnorm": gnorm_show})

        if step % args.eval_interval == 0:
            ev = evaluate("periodic", step)
            if rank == 0:
                log.append(ev)
                with open(out_dir / "probe_log.json", "w") as f:
                    json.dump(log, f, indent=2)

    final = evaluate("final", args.num_steps)
    if rank == 0:
        log.append(final)
        with open(out_dir / "probe_log.json", "w") as f:
            json.dump(log, f, indent=2)

    if args.save_final:
        save_suspect_model(server, out_dir / "model", Path(args.base_model_dir))
    if rank == 0:
        print(f"[done] {time.time()-t0:.1f}s -> {out_dir/'probe_log.json'}", flush=True)


def save_suspect_model(server, model_dir: Path, base_dir: Path):
    """Merge LoRA into the transformer and write a loadable suspect model dir.

    Layout matches assemble_lingbot_descendant: symlink every base component except the
    transformer; write a fresh transformer/ with LoRA folded into plain Linear weights
    (merge_lora_state_dict), so it loads with the stock WanTransformer3DModel used by
    eval_libero_watermark.
    """
    import json as _json
    import torch.distributed as _dist
    from safetensors.torch import save_file
    from wan_va.lora import merge_lora_state_dict

    dist_on = _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1
    rank = _dist.get_rank() if dist_on else 0
    if dist_on:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        sd = get_model_state_dict(
            server.transformer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        sd = server.transformer.state_dict()
    sd = merge_lora_state_dict(sd)
    sd = {k: v.to(torch.bfloat16).cpu() for k, v in sd.items()}
    if rank == 0:
        model_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(base_dir.iterdir()):
            if entry.name == "transformer":
                continue
            link = model_dir / entry.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(entry.resolve())

        tdir = model_dir / "transformer"
        tdir.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(tdir / "diffusion_pytorch_model.safetensors"))
        cfg = dict(server.transformer.config)
        cfg.pop("_name_or_path", None)
        with open(tdir / "config.json", "w") as f:
            _json.dump(cfg, f, indent=2)
        print(f"[save] suspect model -> {model_dir}", flush=True)
    if dist_on:
        _dist.barrier()


if __name__ == "__main__":
    main()
