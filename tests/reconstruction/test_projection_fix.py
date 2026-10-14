import numpy as np

from cbct_n2n.reconstruction.projection_fix import (
    correct_projection_u_offset,
    half_turn_pair_indices,
    residual_column_scores,
)


def test_residual_column_scores_uses_pair_std_then_row_mean():
    first = [
        np.array([[0, 0], [0, 0]], dtype=np.float32),
        np.array([[0, 0], [0, 0]], dtype=np.float32),
    ]
    opposite = [
        np.array([[1, 9], [1, 5]], dtype=np.float32),
        np.array([[3, 5], [3, 9]], dtype=np.float32),
    ]

    scores = residual_column_scores(zip(first, opposite))

    np.testing.assert_allclose(scores, [1.0, 2.0])
    assert int(np.argmin(scores)) == 0


def test_half_turn_pairs_use_configured_scan_angle_range_in_turns():
    pairs = half_turn_pair_indices(1250, angle_range_turns=1.0417)

    assert len(pairs) == 650
    assert pairs[0] == (0, 600)
    assert pairs[-1] == (649, 1249)


def test_half_turn_pairs_for_exact_full_turn_keep_midpoint_pairing():
    pairs = half_turn_pair_indices(1250, angle_range_turns=1.0)

    assert len(pairs) == 625
    assert pairs[0] == (0, 625)
    assert pairs[-1] == (624, 1249)


def test_correct_projection_u_offset_applies_uniform_shift_within_three_mm():
    matrices = np.zeros((3, 21), dtype=np.float32)
    matrices[:, 12:] = np.eye(3, dtype=np.float32).reshape(1, 9)
    original = matrices.copy()

    corrected, info = correct_projection_u_offset(
        matrices, anchor_column=10, origin_column=8, u_spacing_mm=0.5
    )

    np.testing.assert_allclose(corrected[:, 14], original[:, 14] + 2)
    np.testing.assert_array_equal(matrices, original)
    assert info["accepted"] is True
    assert info["offset_px"] == 2
    assert info["distance_mm"] == 1.0


def test_correct_projection_u_offset_rejects_at_three_mm():
    matrices = np.zeros((2, 21), dtype=np.float32)

    corrected, info = correct_projection_u_offset(
        matrices, anchor_column=10, origin_column=4, u_spacing_mm=0.5
    )

    np.testing.assert_array_equal(corrected, matrices)
    assert info["accepted"] is False
    assert info["distance_mm"] == 3.0
