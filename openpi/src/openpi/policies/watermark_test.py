import numpy as np
import pathlib
import sys
import types
import importlib.util
import torch

from openpi.policies import watermark as _watermark


class _EchoNoiseModel(torch.nn.Module):
    def __init__(self, action_horizon: int = 8, action_dim: int = 4):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.last_noise: torch.Tensor | None = None

    def sample_actions(self, device, observation, **kwargs) -> torch.Tensor:  # noqa: ANN003
        noise = kwargs["noise"]
        self.last_noise = noise.detach().cpu().clone()
        return noise


class _AffineActions:
    def __init__(self, scale: float, bias: float):
        self._scale = scale
        self._bias = bias

    def __call__(self, data: dict) -> dict:
        data["actions"] = np.asarray(data["actions"]) * self._scale + self._bias
        return data


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)


def _load_policy_module():
    module_name = "_watermark_policy_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    openpi_mod = types.ModuleType("openpi")
    transforms_mod = types.ModuleType("openpi.transforms")
    transforms_mod.DataTransformFn = object
    transforms_mod.compose = lambda transforms: (lambda data: _apply_transforms(transforms, data))

    models_pkg = types.ModuleType("openpi.models")
    model_mod = types.ModuleType("openpi.models.model")

    class _Observation:
        @classmethod
        def from_dict(cls, data):
            return types.SimpleNamespace(state=data["state"])

    model_mod.Observation = _Observation
    model_mod.BaseModel = object

    shared_pkg = types.ModuleType("openpi.shared")
    array_typing_mod = types.ModuleType("openpi.shared.array_typing")
    array_typing_mod.KeyArrayLike = object
    nnx_utils_mod = types.ModuleType("openpi.shared.nnx_utils")
    nnx_utils_mod.module_jit = lambda fn: fn

    policies_pkg = types.ModuleType("openpi.policies")
    policies_pkg.watermark = _watermark

    base_policy_mod = types.ModuleType("openpi_client.base_policy")
    base_policy_mod.BasePolicy = object

    openpi_client_pkg = types.ModuleType("openpi_client")
    openpi_client_pkg.base_policy = base_policy_mod

    fake_jax = types.ModuleType("jax")
    fake_jax.tree = types.SimpleNamespace(map=_tree_map)
    fake_jax.random = types.SimpleNamespace(split=lambda key, num=2: tuple(range(num)))
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.asarray = np.asarray

    fake_flax = types.ModuleType("flax")
    fake_flax_traverse = types.ModuleType("flax.traverse_util")
    fake_flax.traverse_util = fake_flax_traverse

    openpi_mod.transforms = transforms_mod
    openpi_mod.models = models_pkg
    openpi_mod.shared = shared_pkg
    openpi_mod.policies = policies_pkg
    models_pkg.model = model_mod
    shared_pkg.array_typing = array_typing_mod
    shared_pkg.nnx_utils = nnx_utils_mod

    injected_modules = {
        "openpi": openpi_mod,
        "openpi.transforms": transforms_mod,
        "openpi.models": models_pkg,
        "openpi.models.model": model_mod,
        "openpi.shared": shared_pkg,
        "openpi.shared.array_typing": array_typing_mod,
        "openpi.shared.nnx_utils": nnx_utils_mod,
        "openpi.policies": policies_pkg,
        "openpi.policies.watermark": _watermark,
        "openpi_client": openpi_client_pkg,
        "openpi_client.base_policy": base_policy_mod,
        "jax": fake_jax,
        "jax.numpy": fake_jnp,
        "flax": fake_flax,
        "flax.traverse_util": fake_flax_traverse,
    }

    previous = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)
    try:
        policy_path = pathlib.Path(__file__).with_name("policy.py")
        spec = importlib.util.spec_from_file_location(module_name, policy_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _apply_transforms(transforms, data):
    for transform in transforms:
        data = transform(data)
    return data


def _make_config(**overrides) -> _watermark.InternalNoiseWatermarkConfig:
    config_kwargs = {
        "secret_key": 17,
        "control_freq": 50.0,
        "beta": 0.2,
        "freq_range": (1.0, 3.0),
        "n_tones": 3,
        "watermark_dims": (1, 3),
        "chunk_index_key": "chunk_index",
        "episode_nonce_key": "episode_nonce",
    }
    config_kwargs.update(overrides)
    return _watermark.InternalNoiseWatermarkConfig(**config_kwargs)


def _make_output_config(**overrides) -> _watermark.OutputActionWatermarkConfig:
    config_kwargs = {
        "secret_key": 23,
        "control_freq": 20.0,
        "beta": 0.05,
        "family": "bandpass",
        "watermark_dims": (0, 2),
        "freq_range": (0.8, 2.0),
        "code_type": "balanced_sign",
        "detector": "coherence",
    }
    config_kwargs.update(overrides)
    return _watermark.OutputActionWatermarkConfig(**config_kwargs)


def test_generate_keyed_reference_is_repeatable_and_contextual():
    config = _make_config()
    context = _watermark.WatermarkContext(chunk_index=4, episode_nonce=9)

    ref_a = _watermark.generate_keyed_reference(
        length=16,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    ref_b = _watermark.generate_keyed_reference(
        length=16,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    ref_c = _watermark.generate_keyed_reference(
        length=16,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=_watermark.WatermarkContext(chunk_index=5, episode_nonce=9),
    )

    np.testing.assert_allclose(ref_a, ref_b, rtol=1e-6, atol=1e-7)
    assert ref_a.shape == (16, 4)
    assert not np.allclose(ref_a[:, 1], ref_c[:, 1])


def test_generate_band_passed_gaussian_is_repeatable_and_band_limited():
    signal_a = _watermark._generate_band_passed_gaussian(
        seed=123,
        length=256,
        sample_rate_hz=20.0,
        freq_range=(2.0, 5.0),
    )
    signal_b = _watermark._generate_band_passed_gaussian(
        seed=123,
        length=256,
        sample_rate_hz=20.0,
        freq_range=(2.0, 5.0),
    )

    np.testing.assert_allclose(signal_a, signal_b, rtol=1e-6, atol=1e-7)
    freqs = np.fft.rfftfreq(signal_a.shape[0], d=1.0 / 20.0)
    spectrum = np.abs(np.fft.rfft(signal_a))
    in_band = spectrum[(freqs >= 2.0) & (freqs <= 5.0)]
    out_band = spectrum[(freqs < 1.0) | (freqs > 6.0)]

    assert in_band.mean() > out_band.mean()


def test_mix_internal_noise_only_modifies_target_dims():
    config = _make_config(beta=0.15)
    context = _watermark.WatermarkContext(chunk_index=2, episode_nonce=3)
    base = np.ones((1, 12, 4), dtype=np.float32)

    mixed = _watermark.mix_internal_noise(
        base,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    same = _watermark.mix_internal_noise(
        base,
        sample_rate_hz=config.control_freq,
        config=_make_config(beta=0.0),
        context=context,
    )

    np.testing.assert_allclose(same, base, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(mixed[:, :, 0], base[:, :, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(mixed[:, :, 2], base[:, :, 2], rtol=1e-6, atol=1e-7)
    assert not np.allclose(mixed[:, :, 1], base[:, :, 1])
    assert not np.allclose(mixed[:, :, 3], base[:, :, 3])


def test_generate_keyed_reference_supports_gaussian_mode():
    config = _make_config(reference_mode="gaussian", watermark_dims=(0, 1, 2, 3))
    context = _watermark.WatermarkContext(chunk_index=3, episode_nonce=4)

    ref_a = _watermark.generate_keyed_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    ref_b = _watermark.generate_keyed_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    ref_c = _watermark.generate_keyed_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        config=config,
        context=_watermark.WatermarkContext(chunk_index=4, episode_nonce=4),
    )

    np.testing.assert_allclose(ref_a, ref_b, rtol=1e-6, atol=1e-7)
    assert ref_a.shape == (32, 4)
    assert not np.allclose(ref_a, ref_c)


def test_keyed_chunk_selector_can_skip_injection_for_unselected_chunks():
    config = _make_config(
        beta=0.2,
        reference_mode="gaussian",
        chunk_selection_period=4,
        chunk_selection_count=1,
        watermark_dims=(0, 1, 2, 3),
    )
    selected_context = None
    skipped_context = None
    for chunk_index in range(32):
        context = _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=9)
        if _watermark.should_watermark_chunk(config, context):
            selected_context = context
        else:
            skipped_context = context
        if selected_context is not None and skipped_context is not None:
            break

    assert selected_context is not None
    assert skipped_context is not None

    base = np.ones((1, 12, 4), dtype=np.float32)
    selected = _watermark.mix_internal_noise(
        base,
        sample_rate_hz=config.control_freq,
        config=config,
        context=selected_context,
    )
    skipped = _watermark.mix_internal_noise(
        base,
        sample_rate_hz=config.control_freq,
        config=config,
        context=skipped_context,
    )

    assert not np.allclose(selected, base)
    np.testing.assert_allclose(skipped, base, rtol=1e-6, atol=1e-7)


def test_keyed_chunk_selector_can_choose_exact_fixed_count_per_episode():
    config = _make_config(
        beta=0.2,
        reference_mode="gaussian",
        chunk_selection_strategy="fixed_slots",
        chunk_selection_count=5,
        chunk_selection_total_slots=104,
        watermark_dims=(0, 1, 2, 3),
    )

    selected_indices = [
        chunk_index
        for chunk_index in range(config.chunk_selection_total_slots)
        if _watermark.should_watermark_chunk(
            config,
            _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=9),
        )
    ]
    other_episode_indices = [
        chunk_index
        for chunk_index in range(config.chunk_selection_total_slots)
        if _watermark.should_watermark_chunk(
            config,
            _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=10),
        )
    ]

    assert len(selected_indices) == 5
    assert len(set(selected_indices)) == 5
    assert max(selected_indices) < config.chunk_selection_total_slots
    assert selected_indices != other_episode_indices


def test_stateful_online_chunk_selector_hits_exact_budget_within_bounded_prefix():
    config = _make_config(
        beta=0.2,
        reference_mode="gaussian",
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=10,
        chunk_selection_count=5,
        watermark_dims=(0, 1, 2, 3),
    )

    selected_indices = [
        chunk_index
        for chunk_index in range(64)
        if _watermark.should_watermark_chunk(
            config,
            _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=9),
        )
    ]
    other_episode_indices = [
        chunk_index
        for chunk_index in range(64)
        if _watermark.should_watermark_chunk(
            config,
            _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=10),
        )
    ]

    assert len(selected_indices) == 5
    assert len(set(selected_indices)) == 5
    assert max(selected_indices) < 64
    assert selected_indices != other_episode_indices


def test_policy_infer_watermarks_internal_noise_before_output_transforms():
    policy_module = _load_policy_module()
    config = _make_config(beta=0.25)
    model = _EchoNoiseModel(action_horizon=8, action_dim=4)
    policy = policy_module.Policy(
        model,
        output_transforms=[_AffineActions(scale=3.0, bias=1.0)],
        is_pytorch=True,
        pytorch_device="cpu",
        watermark_config=config,
    )

    obs = {
        "image": {"cam0": np.zeros((8, 8, 3), dtype=np.uint8)},
        "image_mask": {"cam0": np.True_},
        "state": np.zeros((4,), dtype=np.float32),
        "chunk_index": 6,
        "episode_nonce": 13,
    }
    base_noise = np.zeros((8, 4), dtype=np.float32)

    result = policy.infer(obs, noise=base_noise)
    expected_noise = _watermark.mix_internal_noise(
        base_noise[None, ...],
        sample_rate_hz=config.control_freq,
        config=config,
        context=_watermark.WatermarkContext(chunk_index=6, episode_nonce=13),
    )[0]

    assert model.last_noise is not None
    np.testing.assert_allclose(model.last_noise.numpy()[0], expected_noise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["actions"], expected_noise * 3.0 + 1.0, rtol=1e-6, atol=1e-6)


def test_policy_infer_respects_stateful_online_chunk_schedule():
    policy_module = _load_policy_module()
    config = _make_config(
        beta=0.25,
        reference_mode="gaussian",
        chunk_selection_strategy="stateful_online",
        chunk_selection_period=10,
        chunk_selection_count=5,
        watermark_dims=(0, 1, 2, 3),
    )
    model = _EchoNoiseModel(action_horizon=8, action_dim=4)
    policy = policy_module.Policy(
        model,
        is_pytorch=True,
        pytorch_device="cpu",
        watermark_config=config,
    )

    selected_indices = []
    base_noise = np.zeros((8, 4), dtype=np.float32)
    for chunk_index in range(64):
        obs = {
            "image": {"cam0": np.zeros((8, 8, 3), dtype=np.uint8)},
            "image_mask": {"cam0": np.True_},
            "state": np.zeros((4,), dtype=np.float32),
            "chunk_index": chunk_index,
            "episode_nonce": 13,
        }
        result = policy.infer(obs, noise=base_noise)
        if not np.allclose(result["actions"], 0.0):
            selected_indices.append(chunk_index)

    expected_indices = [
        chunk_index
        for chunk_index in range(64)
        if _watermark.should_watermark_chunk(
            config,
            _watermark.WatermarkContext(chunk_index=chunk_index, episode_nonce=13),
        )
    ]

    assert selected_indices == expected_indices
    assert len(selected_indices) == 5


def test_detect_watermark_presence_separates_watermarked_from_plain_traces():
    config = _make_config(beta=0.2)
    rng = np.random.default_rng(0)
    reference = _watermark.generate_reference_trace(
        total_length=256,
        action_dim=4,
        sample_rate_hz=config.control_freq,
        chunk_size=32,
        config=config,
        episode_nonce=5,
    )
    watermarked = reference + 0.35 * rng.standard_normal(reference.shape)
    plain = rng.standard_normal(reference.shape)

    detected = _watermark.detect_watermark_presence(
        watermarked,
        secret_key=config.secret_key,
        sample_rate_hz=config.control_freq,
        chunk_size=32,
        action_dim=4,
        freq_range=config.freq_range,
        n_tones=config.n_tones,
        watermark_dims=config.watermark_dims,
        episode_nonce=5,
        threshold=0.5,
        lag_search_steps=2,
    )
    absent = _watermark.detect_watermark_presence(
        plain,
        secret_key=config.secret_key,
        sample_rate_hz=config.control_freq,
        chunk_size=32,
        action_dim=4,
        freq_range=config.freq_range,
        n_tones=config.n_tones,
        watermark_dims=config.watermark_dims,
        episode_nonce=5,
        threshold=0.5,
        lag_search_steps=2,
    )

    assert detected.score > absent.score
    assert detected.detected
    assert not absent.detected


def test_generate_output_action_reference_supports_bandpass_and_timecode():
    bandpass_config = _make_output_config(family="bandpass", watermark_dims=(0, 1))
    timecode_config = _make_output_config(family="timecode", watermark_dims=(0, 1), detector="matched_filter")
    context = _watermark.WatermarkContext(chunk_index=7, episode_nonce=31)

    bandpass_a = _watermark.generate_output_action_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=bandpass_config.control_freq,
        config=bandpass_config,
        context=context,
    )
    bandpass_b = _watermark.generate_output_action_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=bandpass_config.control_freq,
        config=bandpass_config,
        context=context,
    )
    timecode_a = _watermark.generate_output_action_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=timecode_config.control_freq,
        config=timecode_config,
        context=context,
    )
    timecode_b = _watermark.generate_output_action_reference(
        length=32,
        action_dim=4,
        sample_rate_hz=timecode_config.control_freq,
        config=timecode_config,
        context=_watermark.WatermarkContext(chunk_index=8, episode_nonce=31),
    )

    np.testing.assert_allclose(bandpass_a, bandpass_b, rtol=1e-6, atol=1e-7)
    assert bandpass_a.shape == (32, 4)
    assert timecode_a.shape == (32, 4)
    assert not np.allclose(timecode_a, timecode_b)


def test_apply_output_action_watermark_only_modifies_target_dims_and_tracks_clipping():
    config = _make_output_config(beta=0.2, family="timecode", watermark_dims=(1, 3))
    context = _watermark.WatermarkContext(chunk_index=1, episode_nonce=9)
    base_actions = np.array(
        [
            [0.0, 0.95, 0.0, -0.95],
            [0.0, 0.90, 0.0, -0.90],
            [0.0, 0.85, 0.0, -0.85],
        ],
        dtype=np.float32,
    )

    applied = _watermark.apply_output_action_watermark(
        base_actions,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
        clip_low=-1.0,
        clip_high=1.0,
    )

    np.testing.assert_allclose(applied.watermarked_actions[:, 0], base_actions[:, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(applied.watermarked_actions[:, 2], base_actions[:, 2], rtol=1e-6, atol=1e-7)
    assert not np.allclose(applied.watermarked_actions[:, 1], base_actions[:, 1])
    assert not np.allclose(applied.watermarked_actions[:, 3], base_actions[:, 3])
    assert applied.reference.shape == base_actions.shape
    assert applied.clip_fraction > 0.0
    assert applied.delta_rms > 0.0


def test_output_action_timecode_detectors_prefer_matching_reference():
    config = _make_output_config(
        family="timecode",
        detector="matched_filter",
        beta=0.05,
        watermark_dims=(0, 1, 2),
    )
    context = _watermark.WatermarkContext(chunk_index=3, episode_nonce=11)
    reference = _watermark.generate_output_action_reference(
        length=128,
        action_dim=3,
        sample_rate_hz=config.control_freq,
        config=config,
        context=context,
    )
    rng = np.random.default_rng(7)
    telemetry = reference + 0.25 * rng.standard_normal(reference.shape).astype(np.float32)
    wrong_reference = _watermark.generate_output_action_reference(
        length=128,
        action_dim=3,
        sample_rate_hz=config.control_freq,
        config=config,
        context=_watermark.WatermarkContext(chunk_index=4, episode_nonce=11),
    )

    matched_score, _ = _watermark.matched_filter_score(
        telemetry,
        reference,
        watermark_dims=(0, 1, 2),
    )
    matched_wrong, _ = _watermark.matched_filter_score(
        telemetry,
        wrong_reference,
        watermark_dims=(0, 1, 2),
    )
    glrt_score, _ = _watermark.glrt_score(
        telemetry,
        reference,
        watermark_dims=(0, 1, 2),
    )
    glrt_wrong, _ = _watermark.glrt_score(
        telemetry,
        wrong_reference,
        watermark_dims=(0, 1, 2),
    )

    assert matched_score > matched_wrong
    assert glrt_score > glrt_wrong
