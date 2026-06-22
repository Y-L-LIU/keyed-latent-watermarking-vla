"""Toy exact-equivalence test for batched-vs-serial multi-start MAP (JAX optimizer).

Proves: optimizing the SUM of per-restart energies on a stacked latent gives results
bit-identical to optimizing each restart alone. Run: JAX_PLATFORMS=cpu python3 this_file.py
(Used to validate openpi/scripts/eval_robotwin_watermark_map.py; gave max|diff|=0.0.)

For the PyTorch targets (lingbot fm_latent_map_solver.run_map_restarts, openpi
eval_libero_action_inversion._run_pytorch_channel_latent_map_restarts), write the analogous
test with torch.optim.Adam: stack inits along batch, loss = per-restart-energy.sum().
"""
import jax, jax.numpy as jnp, numpy as np


def opt(init_noise, loss_fn, num_steps, learning_rate):  # verbatim _optimize_latent_with_adam_jax
    b1, b2, eps = 0.9, 0.999, 1e-8
    lvg = jax.value_and_grad(loss_fn)
    def step(i, carry):
        lat, m, v = carry
        _, g = lvg(lat)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * jnp.square(g)
        s = jnp.asarray(i + 1, dtype=jnp.float32)
        lat = lat - learning_rate * (m / (1 - b1 ** s)) / (jnp.sqrt(v / (1 - b2 ** s)) + eps)
        return lat, m, v
    lat, _, _ = jax.lax.fori_loop(0, int(num_steps), step,
                                  (init_noise, jnp.zeros_like(init_noise), jnp.zeros_like(init_noise)))
    return lat


B, H, D = 4, 5, 32
rng = np.random.default_rng(123)
z0 = rng.standard_normal((B, H, D)).astype(np.float32)        # same draw order as serial
T = jnp.asarray(rng.standard_normal((B, H, D)).astype(np.float32))
per_sample = lambda z: jnp.mean(jnp.square(z - T), axis=(1, 2))  # (B,)

zb = opt(jnp.asarray(z0), lambda z: jnp.sum(per_sample(z)), 100, 0.1)         # BATCHED
seq = np.stack([np.asarray(opt(jnp.asarray(z0[i:i+1]),
                              lambda z, i=i: jnp.mean(jnp.square(z - T[i:i+1])), 100, 0.1)[0])
                for i in range(B)])                                            # SERIAL
dz = float(np.max(np.abs(np.asarray(zb) - seq)))
print("max|z_batched - z_serial| =", dz)
print("EQUIVALENT" if dz < 1e-4 else "MISMATCH!!")
