"""Correctness tests for the 3D guided filter and its sharpening modes.

The filter is only worth trusting if it agrees with the textbook (per-window
regression) definition and if the slabbed streaming version agrees with the
whole-volume one, so both are checked against explicit references rather than
against a second call of the same code.
"""
from __future__ import annotations

import numpy as np
import pytest

from cbct_n2n.filters.guided import (
    box_mean_3d,
    cap_darkening,
    eps_from_noise,
    guided_filter_3d,
    guided_sharpen_3d,
    guided_structure_field,
    iter_z_slabs,
)


def _reference_guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float):
    """Textbook O(N * (2r+1)^3) guided filter, including the second box mean.

    Boundary handling is ``mode="nearest"`` i.e. edge replication, which is what
    ``np.pad(..., mode="edge")`` gives.
    """
    radius = int(radius)
    gp = np.pad(np.asarray(guide, np.float64), radius, mode="edge")
    sp = np.pad(np.asarray(src, np.float64), radius, mode="edge")
    shape = guide.shape
    size = 2 * radius + 1
    a = np.empty(shape)
    b = np.empty(shape)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                wg = gp[z:z + size, y:y + size, x:x + size]
                ws = sp[z:z + size, y:y + size, x:x + size]
                mean_g, mean_s = wg.mean(), ws.mean()
                dg, ds = wg - mean_g, ws - mean_s
                var_g = float((dg * dg).mean())
                cov = float((dg * ds).mean())
                a[z, y, x] = cov / (var_g + eps)
                b[z, y, x] = mean_s - a[z, y, x] * mean_g
    ap, bp = np.pad(a, radius, mode="edge"), np.pad(b, radius, mode="edge")
    out = np.empty(shape)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                out[z, y, x] = (
                    ap[z:z + size, y:y + size, x:x + size].mean() * guide[z, y, x]
                    + bp[z:z + size, y:y + size, x:x + size].mean()
                )
    return out


def _smooth_field(shape=(10, 9, 8), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.normal(size=shape).astype(np.float32)
    return box_mean_3d(box_mean_3d(coarse, 2), 2)


# --------------------------------------------------------------------------- box mean


def test_box_mean_matches_edge_padded_window_mean():
    rng = np.random.default_rng(1)
    volume = rng.normal(size=(9, 7, 6)).astype(np.float32)
    padded = np.pad(volume, 2, mode="edge")
    expected = np.empty_like(volume)
    for z in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                expected[z, y, x] = padded[z:z + 5, y:y + 5, x:x + 5].mean()
    np.testing.assert_allclose(box_mean_3d(volume, 2), expected, atol=1e-6)


def test_box_mean_preserves_float32_and_shape():
    volume = np.zeros((5, 4, 3), dtype=np.float32)
    out = box_mean_3d(volume, 1)
    assert out.dtype == np.float32 and out.shape == volume.shape


def test_box_mean_rejects_bad_input():
    with pytest.raises(ValueError):
        box_mean_3d(np.zeros((4, 4), dtype=np.float32), 1)
    with pytest.raises(ValueError):
        box_mean_3d(np.zeros((4, 4, 4), dtype=np.float32), 0)


# --------------------------------------------------------------------------- guided filter


def test_matches_the_textbook_per_window_filter():
    guide = _smooth_field()
    src = _smooth_field(seed=7)
    expected = _reference_guided_filter(guide, src, radius=2, eps=0.01)
    result = guided_filter_3d(guide, src, radius=2, eps=0.01)
    np.testing.assert_allclose(result.base, expected, atol=2e-6)


def test_self_guidance_matches_joint_guidance_with_the_same_guide():
    volume = _smooth_field()
    np.testing.assert_array_equal(
        guided_filter_3d(volume, radius=1, eps=0.05).base,
        guided_filter_3d(volume, volume, radius=1, eps=0.05).base,
    )


def test_tiny_eps_approaches_the_identity_on_a_structured_field():
    volume = _smooth_field()
    base = guided_filter_3d(volume, radius=2, eps=1e-12).base
    assert float(np.abs(base - volume).max()) < 1e-4


def test_huge_eps_approaches_a_double_box_mean():
    volume = _smooth_field()
    base = guided_filter_3d(volume, radius=2, eps=1e12).base
    double = box_mean_3d(box_mean_3d(volume, 2), 2)
    np.testing.assert_allclose(base, double, atol=1e-6)


def test_wiener_gain_stays_in_the_unit_interval():
    volume = _smooth_field()
    a = guided_filter_3d(volume, radius=1, eps=0.05, keep_a=True).a
    assert a is not None
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_the_local_mean_is_the_source_mean_not_the_guide_mean():
    volume = _smooth_field()
    guide = _smooth_field(seed=3)
    result = guided_filter_3d(guide, volume, radius=2, eps=0.05)
    np.testing.assert_allclose(result.mean_src, box_mean_3d(volume, 2), atol=1e-6)


# --------------------------------------------------------------------------- slabbing


@pytest.mark.parametrize("slab", [1, 3, 4, 5, 7, 16])
def test_slabbed_streaming_equals_whole_volume(slab: int):
    volume = _smooth_field(shape=(11, 9, 8))
    whole = guided_filter_3d(volume, radius=3, eps=0.02).base
    streamed = guided_filter_3d(volume, radius=3, eps=0.02, slab=slab).base
    np.testing.assert_allclose(streamed, whole, atol=1e-6)


def test_iter_z_slabs_clips_the_halo_at_the_volume_ends():
    slabs = list(iter_z_slabs(nz=10, slab=4, halo=6))
    assert [(z0, z1) for z0, z1, _, _ in slabs] == [(0, 4), (4, 8), (8, 10)]
    assert [(gz0, gz1) for _, _, gz0, gz1 in slabs] == [(0, 10), (0, 10), (2, 10)]


def test_on_slab_callback_walks_the_volume_once():
    seen: list[tuple[int, int]] = []
    guided_filter_3d(
        _smooth_field(shape=(9, 5, 5)), radius=1, eps=0.1, slab=4,
        on_slab=lambda z0, z1: seen.append((z0, z1)),
    )
    assert seen == [(0, 4), (4, 8), (8, 9)]


# --------------------------------------------------------------------------- sharpening


def test_strength_zero_returns_the_input_bit_for_bit():
    volume = _smooth_field()
    out = guided_sharpen_3d(volume, radius=2, eps=0.01, strength=0.0)
    np.testing.assert_array_equal(out, volume)


def test_wiener_structure_is_the_base_minus_the_local_mean():
    volume = _smooth_field()
    result = guided_filter_3d(volume, radius=2, eps=0.01)
    field = guided_structure_field(volume, radius=2, eps=0.01, mode="wiener")
    np.testing.assert_array_equal(field, result.structure)


def test_detail_field_is_the_guided_residual():
    volume = _smooth_field()
    result = guided_filter_3d(volume, radius=2, eps=0.01)
    field = guided_structure_field(volume, radius=2, eps=0.01, mode="detail")
    np.testing.assert_allclose(field, volume - result.base, atol=1e-7)


def test_structure_field_is_linear_in_strength():
    volume = _smooth_field()
    one = guided_sharpen_3d(volume, radius=2, eps=0.01, strength=1.0)
    two = guided_sharpen_3d(volume, radius=2, eps=0.01, strength=2.0)
    np.testing.assert_allclose(two, volume + 2.0 * (one - volume), atol=1e-6)


def test_sharpening_steepens_a_step_edge():
    volume = np.zeros((5, 9, 9), dtype=np.float32)
    volume[:, :, 4:] = 1.0
    volume = box_mean_3d(volume, 1)  # a 1-voxel-wide ramp instead of an ideal jump
    before = np.abs(np.diff(volume[2, 4, :]))

    for mode in ("wiener", "detail"):
        # eps must be comparable to the edge's own local variance (0.07) or the
        # guided filter is a near-identity and there is nothing to boost.
        out = guided_sharpen_3d(volume, radius=2, eps=0.05, strength=1.0, mode=mode)
        after = np.abs(np.diff(out[2, 4, :]))
        assert after.max() > before.max(), mode


def test_a_flat_region_is_left_bit_for_bit_alone():
    volume = np.zeros((11, 21, 21), dtype=np.float32)
    volume[:, :, 14:] = 1.0
    volume = box_mean_3d(volume, 1)  # the ramp now starts at x=13
    for mode in ("wiener", "detail"):
        out = guided_sharpen_3d(volume, radius=2, eps=0.05, strength=2.0, mode=mode)
        # Both box means have support 2r = 4, so a voxel more than 4 + r = 6
        # voxels away from x=13 sees neither the gain A nor the offset B.
        assert out[5, 10, 0] == volume[5, 10, 0], mode
        assert out[5, 10, 5] == volume[5, 10, 5], mode


def test_flats_next_to_a_structure_are_not_spared_and_that_is_by_design():
    # A = box_r(a) is deliberately averaged over the window, so the gain leaks r
    # voxels out of a structure.  Without that smoothing the filter would show
    # blocking seams at the window grid; with it, a bone edge also nudges the
    # soft tissue next to it -- which is what the k knob has to bound.
    volume = np.zeros((11, 21, 21), dtype=np.float32)
    volume[:, :, 14:] = 1.0
    volume = box_mean_3d(volume, 1)
    out = guided_sharpen_3d(volume, radius=2, eps=0.05, strength=2.0, mode="wiener")
    assert abs(float(out[5, 10, 9])) > 1e-6


def _noise_gain(volume: np.ndarray, **kwargs) -> float:
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    out = guided_sharpen_3d(volume, **kwargs)
    return float(out[interior].std()) / float(volume[interior].std())


def test_noise_gain_of_the_wiener_mode_is_one_plus_one_over_one_plus_k_squared():
    rng = np.random.default_rng(11)
    sigma = 0.01
    noise = (rng.normal(scale=sigma, size=(20, 40, 40)) + 0.2).astype(np.float32)
    for k, expected in ((2.0, 1 + 1 / 5), (4.0, 1 + 1 / 17)):
        gain = _noise_gain(
            noise, radius=2, eps=eps_from_noise(sigma, k=k), strength=1.0, mode="wiener"
        )
        # In a noise-only field the local Wiener gain is a = 1/(1+k^2)
        # everywhere, so the boost adds s*a*(-) to the fluctuation instead of
        # s*(-); the measured std gain has to land on 1 + s/(1+k^2).
        assert gain == pytest.approx(expected, abs=0.02), k


def test_classic_detail_mode_pays_the_full_noise_price():
    rng = np.random.default_rng(11)
    sigma = 0.01
    noise = (rng.normal(scale=sigma, size=(20, 40, 40)) + 0.2).astype(np.float32)
    eps = eps_from_noise(sigma, k=2.0)
    wiener = _noise_gain(noise, radius=2, eps=eps, strength=1.0, mode="wiener")
    detail = _noise_gain(noise, radius=2, eps=eps, strength=1.0, mode="detail")
    # detail boosts the whole residual: gain roughly 1 + s*(1 - a) = 1.8.
    assert detail > 1.70
    assert wiener < 1.25
    assert detail > 1.4 * wiener
    # ... and the wiener gain is monotone in k: more k, less noise penalty.
    quieter = _noise_gain(
        noise, radius=2, eps=eps_from_noise(sigma, k=6.0), strength=1.0, mode="wiener"
    )
    assert quieter < wiener


def test_both_modes_are_mean_preserving():
    rng = np.random.default_rng(5)
    volume = (rng.normal(scale=0.01, size=(8, 24, 24)) + 0.2).astype(np.float32)
    eps = eps_from_noise(0.01, k=2.0)
    for mode in ("wiener", "detail"):
        out = guided_sharpen_3d(volume, radius=2, eps=eps, strength=2.0, mode=mode)
        # The boost is a high-pass: it moves no DC (round-off only).
        assert abs(float(out.mean() - volume.mean())) < 5e-4, mode


def test_joint_guidance_changes_the_result_and_honours_the_guide():
    src = _smooth_field()
    sharp_guide = np.zeros_like(src)
    sharp_guide[:, :, 0::2] = 1.0  # a structure the source does not have
    flat = guided_sharpen_3d(src, radius=2, eps=0.01, strength=1.0)
    joint = guided_sharpen_3d(src, radius=2, eps=0.01, strength=1.0, guide=sharp_guide)
    assert not np.allclose(flat, joint)
    identical = guided_sharpen_3d(src, radius=2, eps=0.01, strength=1.0, guide=src)
    np.testing.assert_allclose(identical, flat, atol=1e-7)


def test_out_buffer_is_honoured():
    volume = _smooth_field()
    buffer = np.empty_like(volume)
    out = guided_sharpen_3d(volume, radius=1, eps=0.01, strength=1.5, out=buffer)
    assert out is buffer


# --------------------------------------------------------------------------- validation


def test_eps_from_noise_scales_with_sigma_and_k():
    assert eps_from_noise(0.01, k=2.0) == pytest.approx(4e-4)
    assert eps_from_noise(0.01, k=4.0) == pytest.approx(1.6e-3)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            eps_from_noise(bad)
    with pytest.raises(ValueError):
        eps_from_noise(0.01, k=0.0)


def test_rejects_bad_arguments():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        guided_filter_3d(volume, radius=0, eps=0.1)
    with pytest.raises(ValueError):
        guided_filter_3d(volume, radius=1, eps=0.0)
    with pytest.raises(ValueError):
        guided_filter_3d(volume, np.zeros((4, 4, 3), dtype=np.float32), radius=1, eps=0.1)
    with pytest.raises(ValueError):
        guided_filter_3d(np.zeros((4, 4), dtype=np.float32), radius=1, eps=0.1)
    with pytest.raises(ValueError):
        guided_structure_field(volume, radius=1, eps=0.1, mode="nonsense")
    with pytest.raises(ValueError):
        guided_structure_field(volume, radius=1, eps=0.1, mode="detail", guide=volume)
    with pytest.raises(ValueError):
        guided_sharpen_3d(volume, radius=1, eps=0.1, out=np.empty((4, 4, 3), np.float32))


# --------------------------------------------------------------------- darkening cap
#
# The guided field is ``q - m ~= A*(p - m)``, so where A -> 1 (a strong edge) it
# degenerates into unsharp masking with a window of radius r, and a low-density
# gap narrower than that window gets its floor pulled down by the dense walls.
# That is the dark rim.  `cap_darkening` bounds it; eps/k provably cannot
# (measured: k=4/8/12 give the same worst darkening to 3 decimals).


def test_cap_is_off_when_max_darken_is_none():
    volume = np.array([0.0, 0.3, 0.6], dtype=np.float32)
    sharp = np.array([-0.2, 0.3, 0.9], dtype=np.float32)
    assert cap_darkening(sharp, volume, None) is sharp


def test_cap_leaves_the_bright_side_bit_for_bit_alone():
    """One-sided on purpose: brightening trabeculae is half the gain."""
    volume = np.full((3, 3, 3), 0.3, dtype=np.float32)
    sharp = volume.copy()
    sharp[0, 0, 0] = 0.9
    sharp[1, 1, 1] = 0.4
    out = cap_darkening(sharp, volume, 0.1)
    assert out[0, 0, 0] == sharp[0, 0, 0]
    assert out[1, 1, 1] == sharp[1, 1, 1]


def test_cap_is_the_identity_where_nothing_darkened():
    volume = np.full((3, 3, 3), 0.3, dtype=np.float32)
    out = cap_darkening(volume.copy(), volume, 0.05)
    assert np.array_equal(out, volume)


def test_cap_bounds_the_darkening_at_max_darken():
    volume = np.full(6, 0.5, dtype=np.float32)
    # Darkenings from tiny to absurd, including far past the cap.
    darkening = np.array([1e-6, 1e-3, 0.01, 0.05, 0.2, 2.0], dtype=np.float32)
    sharp = volume - darkening
    for cap in (0.05, 0.2):
        out = cap_darkening(sharp, volume, cap)
        depth = volume - out
        assert (depth <= cap + 1e-6).all()
        assert np.all(depth >= 0)
        # ... and it actually binds on the big ones.
        assert depth[-1] == pytest.approx(cap, abs=1e-5)


def test_cap_is_transparent_when_the_darkening_is_small():
    """f'(0) = 1, so a gentle gradient is not kinked into a step.

    Without this the clamp would behave like a threshold and post-contrast the
    volume wherever the cap happens to bind.
    """
    volume = np.full(4, 0.5, dtype=np.float32)
    darkening = np.array([0.0, 1e-5, 1e-4, 1e-3], dtype=np.float32)
    sharp = volume - darkening
    out = cap_darkening(sharp, volume, 0.2)
    assert np.allclose(out, sharp, atol=1e-6)


def test_cap_preserves_order():
    """A monotone profile must stay monotone: the correction is monotone in the
    overshoot, so no inversion (which would read as a weird edge) is created."""
    volume = np.linspace(0.2, 0.5, 64, dtype=np.float32)
    ramp = np.linspace(0.9, 0.1, 64, dtype=np.float32)      # deepening overshoot
    out = cap_darkening(volume - ramp, volume, 0.1)
    assert (np.diff(out) >= -1e-7).all()


def test_cap_rejects_a_non_positive_or_non_finite_max_darken():
    volume = np.zeros(4, dtype=np.float32)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cap_darkening(volume, volume, bad)


def test_guided_sharpen_accepts_a_max_darken_bound():
    """The bound reaches through the public entry point, and out = in at s = 0."""
    rng = np.random.default_rng(3)
    volume = np.full((12, 16, 16), 0.3, dtype=np.float32)
    volume += rng.normal(0.0, 0.01, size=volume.shape).astype(np.float32)
    volume[6:, 8, 8] = 0.9                                   # a step to ring against

    free = guided_sharpen_3d(volume, radius=2, eps=1e-5, strength=1.0)
    bound = guided_sharpen_3d(volume, radius=2, eps=1e-5, strength=1.0, max_darken=0.05)
    assert (volume - bound).max() <= 0.05 + 1e-6
    assert (volume - free).max() > 0.05                      # the cap was needed
    assert (bound >= volume).sum() >= (free >= volume).sum()  # bright side intact

    identity = guided_sharpen_3d(volume, radius=2, eps=1e-5, strength=0.0, max_darken=0.05)
    assert np.array_equal(identity, volume)
