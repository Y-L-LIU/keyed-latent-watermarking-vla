from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import watermark as _watermark
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        watermark_config: _watermark.InternalNoiseWatermarkConfig | None = None,
        watermark_global_step_key: str = "global_step",
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            watermark_config: Optional configuration for internal watermark mixing on sampler noise.
            watermark_global_step_key: Backwards-compatible fallback observation key used as chunk index when a
                dedicated chunk index is not available.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._watermark_config = watermark_config
        self._watermark_global_step_key = watermark_global_step_key

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        watermark_context = self._extract_watermark_context(obs)
        obs = self._strip_runtime_metadata(obs)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            provided_noise = noise if noise is not None else self._sample_kwargs.get("noise")
            if self._watermark_config is not None and provided_noise is None:
                self._rng, sample_rng_or_pytorch_device, noise_rng = jax.random.split(self._rng, 3)
            else:
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
                noise_rng = None
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
            noise_rng = None

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        provided_noise = noise if noise is not None else sample_kwargs.pop("noise", None)
        prepared_noise = self._prepare_internal_noise(
            provided_noise,
            batch_size=inputs["state"].shape[0],
            sample_rng_or_pytorch_device=sample_rng_or_pytorch_device,
            noise_rng=noise_rng,
            context=watermark_context,
        )
        if prepared_noise is not None:
            sample_kwargs["noise"] = prepared_noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Watermarking is intentionally confined to the sampler noise prepared
        # above. Nothing additive is applied after decode or output transforms.
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _extract_watermark_context(self, obs: dict) -> _watermark.WatermarkContext | None:
        if self._watermark_config is None:
            return None
        cfg = self._watermark_config
        assert cfg is not None
        chunk_index = self._extract_optional_scalar(obs, cfg.chunk_index_key)
        if chunk_index is None and cfg.fallback_chunk_index_key is not None:
            chunk_index = self._extract_optional_scalar(obs, cfg.fallback_chunk_index_key)
        if chunk_index is None:
            chunk_index = self._extract_optional_scalar(obs, self._watermark_global_step_key)
        if chunk_index is None:
            chunk_index = 0

        episode_nonce = 0
        if cfg.episode_nonce_key is not None:
            maybe_episode_nonce = self._extract_optional_scalar(obs, cfg.episode_nonce_key)
            if maybe_episode_nonce is not None:
                episode_nonce = maybe_episode_nonce

        obs_seed = None
        if getattr(cfg, "keying_mode", "nonce") == "observation":
            feature = obs.get(cfg.obs_key)
            if feature is None:
                raise ValueError(
                    f"watermark keying_mode='observation' requires obs[{cfg.obs_key!r}]; "
                    f"available keys: {sorted(obs.keys())}"
                )
            obs_seed = _watermark.compute_obs_seed(
                np.asarray(feature),
                quantization=cfg.obs_quantization,
                proj_dims=cfg.obs_proj_dims,
            )

        return _watermark.WatermarkContext(
            chunk_index=chunk_index, episode_nonce=episode_nonce, obs_seed=obs_seed
        )

    def _strip_runtime_metadata(self, obs: dict) -> dict:
        if self._watermark_config is None:
            return obs

        cfg = self._watermark_config
        reserved_keys = {self._watermark_global_step_key, cfg.chunk_index_key}
        if cfg.fallback_chunk_index_key is not None:
            reserved_keys.add(cfg.fallback_chunk_index_key)
        if cfg.episode_nonce_key is not None:
            reserved_keys.add(cfg.episode_nonce_key)
        return {key: value for key, value in obs.items() if key not in reserved_keys}

    def _extract_optional_scalar(self, obs: dict, key: str | None) -> int | None:
        if key is None or key not in obs:
            return None
        try:
            return int(np.asarray(obs[key]).item())
        except Exception as exc:
            raise ValueError(f"Expected scalar '{key}', got {obs[key]!r}.") from exc

    def _prepare_internal_noise(
        self,
        noise,
        *,
        batch_size: int,
        sample_rng_or_pytorch_device,
        noise_rng,
        context: _watermark.WatermarkContext | None,
    ):
        if noise is None and self._watermark_config is None:
            return None

        base_noise = self._coerce_or_sample_noise(
            noise,
            batch_size=batch_size,
            noise_rng=noise_rng,
        )
        if self._watermark_config is None:
            return base_noise

        cfg = self._watermark_config
        assert cfg is not None
        return _watermark.mix_internal_noise(
            base_noise,
            sample_rate_hz=cfg.control_freq,
            config=cfg,
            context=context or _watermark.WatermarkContext(),
        )

    def _coerce_or_sample_noise(
        self,
        noise,
        *,
        batch_size: int,
        noise_rng,
    ):
        if noise is None:
            shape = (batch_size, self._model.action_horizon, self._model.action_dim)
            if self._is_pytorch_model:
                return torch.randn(shape, dtype=torch.float32, device=self._pytorch_device)
            if noise_rng is None:
                raise ValueError("noise_rng is required when sampling internal watermark noise for JAX models.")
            return jax.random.normal(noise_rng, shape, dtype=jnp.float32)

        if self._is_pytorch_model:
            noise = noise if isinstance(noise, torch.Tensor) else torch.from_numpy(np.asarray(noise))
            noise = noise.to(self._pytorch_device, dtype=torch.float32)
        else:
            noise = jnp.asarray(noise, dtype=jnp.float32)

        if noise.ndim == 2:
            noise = noise[None, ...]
        if noise.ndim != 3:
            raise ValueError(f"Expected internal noise with rank 2 or 3, got shape={noise.shape}")
        return noise

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
