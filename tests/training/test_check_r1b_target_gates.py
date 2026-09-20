"""Unit tests for scripts/check_r1b_target_gates.py (blend-at-round-1 gate)."""

from __future__ import annotations

import json

from scripts.check_r1b_target_gates import (
    BANDS,
    evaluate_gates,
    evaluate_prep_gate,
    main,
    prep_coverage,
    prep_gate_warnings,
)


def _prep(**overrides) -> dict:
    base = {
        "n_new": 200,
        "n_existing": 0,
        "n_skip_missing": 0,
        "noise_immerkaer": {
            "full_mean": 0.0160,
            "dn_mean": 0.0100,
            "delta_mean": 0.0060,
            "relative_reduction_mean": 0.375,
            "n": 200,
        },
    }
    base.update(overrides)
    return base


def _blend(**overrides) -> dict:
    # w = 0 in the low band -> target == full ; w = 1 in the high band -> target == M0
    base = {
        "n_new": 200,
        "n_existing": 0,
        "n_skip_missing": 0,
        "mask_source": "high",
        "lo": 0.2,
        "hi": 0.4,
        "bands": {
            "low": {"full": 0.0100, "r1": 0.0100, "r2": 0.0060, "target": 0.0100, "n": 200},
            "mid": {"full": 0.0150, "r1": 0.0150, "r2": 0.0090, "target": 0.0120, "n": 200},
            "high": {"full": 0.0200, "r1": 0.0200, "r2": 0.0110, "target": 0.0110, "n": 200},
        },
    }
    base.update(overrides)
    return base


def test_gate_passes_on_well_formed_overviews():
    assert evaluate_gates(_prep(), _blend()) == []


def test_band_order_is_soft_tissue_first():
    assert [band for band, _ in BANDS] == ["low", "mid", "high"]


def test_gate_fails_when_teacher_does_not_clean_full():
    failures = evaluate_gates(_prep(noise_immerkaer={"delta_mean": -0.001,
                                                    "relative_reduction_mean": -0.05}), _blend())
    assert any("did not clean full" in line for line in failures)


def test_gate_fails_when_reduction_below_threshold():
    failures = evaluate_gates(_prep(noise_immerkaer={"delta_mean": 0.0002,
                                                    "relative_reduction_mean": 0.01}), _blend())
    assert any("relative reduction" in line for line in failures)


def test_gate_fails_on_skipped_or_incomplete_coverage():
    failures = evaluate_gates(_prep(n_skip_missing=3), _blend(n_new=100))
    assert any("stage A skipped" in line for line in failures)
    assert any("stage B covered 100 studies" in line for line in failures)


def test_gate_fails_when_blend_key_is_the_noisy_full():
    failures = evaluate_gates(_prep(), _blend(mask_source="low"))
    assert any("mask_source" in line for line in failures)


def test_gate_fails_when_low_band_target_is_not_full():
    blend = _blend()
    blend["bands"]["low"]["target"] = 0.0080  # leaked M0 output into the soft-tissue band
    failures = evaluate_gates(_prep(), blend)
    assert any("low band target" in line for line in failures)


def test_gate_fails_when_high_band_target_is_not_m0():
    blend = _blend()
    blend["bands"]["high"]["target"] = 0.0180  # bone kept the noisy full instead of M0
    failures = evaluate_gates(_prep(), blend)
    assert any("high band target" in line for line in failures)


def test_gate_fails_when_m0_is_noisier_than_full_in_bone():
    blend = _blend()
    blend["bands"]["high"].update({"full": 0.0110, "r2": 0.0200, "target": 0.0200})
    failures = evaluate_gates(_prep(), blend)
    assert any("not cleaner than full" in line for line in failures)


def test_gate_fails_when_mid_band_leaves_the_parent_range():
    blend = _blend()
    blend["bands"]["mid"]["target"] = 0.0200  # above both parents
    failures = evaluate_gates(_prep(), blend)
    assert any("outside [full, M0]" in line for line in failures)


def test_main_writes_nothing_and_reports_pass(tmp_path, capsys):
    prep_path = tmp_path / "prepare_full_dn_3d_128b_r0_overview.json"
    blend_path = tmp_path / "blend_full_dn_3d_128b_r1_overview.json"
    prep_path.write_text(json.dumps(_prep()), encoding="utf-8")
    blend_path.write_text(json.dumps(_blend()), encoding="utf-8")

    import sys

    argv = sys.argv
    sys.argv = ["check_r1b_target_gates.py", "--prep-overview", str(prep_path),
                "--blend-overview", str(blend_path), "--expect-studies", "200"]
    try:
        code = main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert code == 0
    assert "GATE RESULT: PASS" in out
    # the gate is read-only
    assert json.loads(prep_path.read_text(encoding="utf-8")) == _prep()


def test_main_exits_nonzero_on_failure(tmp_path, capsys):
    prep_path = tmp_path / "prep.json"
    blend_path = tmp_path / "blend.json"
    prep_path.write_text(json.dumps(_prep(n_skip_missing=5)), encoding="utf-8")
    blend_path.write_text(json.dumps(_blend()), encoding="utf-8")

    import sys

    argv = sys.argv
    sys.argv = ["check_r1b_target_gates.py", "--prep-overview", str(prep_path),
                "--blend-overview", str(blend_path), "--expect-studies", "200"]
    try:
        code = main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert code == 1
    assert "GATE RESULT: FAIL" in out


# --- plain-recursion rounds (r2b / r3b / r4b): prep-only gate -----------------


def test_prep_only_gate_passes_without_a_blend_overview():
    assert evaluate_prep_gate(_prep()) == []


def test_prep_only_main_runs_without_blend_overview(tmp_path, capsys):
    prep_path = tmp_path / "prepare_full_dn_3d_128b_r2_overview.json"
    prep_path.write_text(json.dumps(_prep()), encoding="utf-8")

    import sys

    argv = sys.argv
    sys.argv = ["check_r1b_target_gates.py", "--prep-overview", str(prep_path),
                "--expect-studies", "200", "--label", "r2b"]
    try:
        code = main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert code == 0
    assert "GATE RESULT: PASS -- safe to start r2b" in out
    assert "STAGE B -- skipped" in out


def test_prep_only_main_fails_on_a_degenerate_target(tmp_path, capsys):
    prep_path = tmp_path / "prepare_full_dn_3d_128b_r3_overview.json"
    prep_path.write_text(json.dumps(_prep(noise_immerkaer={
        "delta_mean": 0.0001, "relative_reduction_mean": 0.004, "n": 200})), encoding="utf-8")

    import sys

    argv = sys.argv
    sys.argv = ["check_r1b_target_gates.py", "--prep-overview", str(prep_path)]
    try:
        code = main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert code == 1
    assert "relative reduction" in out


def test_prep_coverage_prefers_merged_rows_over_run_local_counts():
    """A resumed generation reports few new files but must still cover 200 studies."""
    prep = _prep(n_new=5, n_existing=0, rows=[{"key": f"k{i}"} for i in range(200)])
    assert prep_coverage(prep) == 200
    assert evaluate_prep_gate(prep) == []


def test_prep_gate_warns_but_does_not_fail_on_suspicious_reduction():
    prep = _prep(noise_immerkaer={"full_mean": 0.016, "dn_mean": 0.003,
                                  "delta_mean": 0.013, "relative_reduction_mean": 0.81,
                                  "n": 200})
    assert evaluate_prep_gate(prep) == []
    warnings = prep_gate_warnings(prep)
    assert warnings and "over-smoothing" in warnings[0]
