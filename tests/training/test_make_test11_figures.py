"""Unit tests for scripts/make_test11_figures.py path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.make_test11_figures import (
    _input_of,
    figure_name,
    render_noise_summary,
    render_volume_case,
)


def test_figure_name_disambiguates_studies_of_the_same_device(tmp_path):
    """Mode3_1609 has three studies on one device; names must not collide."""
    common = {"group": "Mode3_1609", "case_id": "YPDL-26A-0026", "domain": "full.raw"}
    first = dict(common, study_uid="1.2.156.89797079.835519821.1189836059.1130746797",
                 output=str(tmp_path / "denoised" / "full_denoised.raw"))
    second = dict(common, study_uid="1.2.826.0.1.3680043.2.461.16167459.1916027858",
                  output=str(tmp_path / "denoised" / "full_denoised.raw"))
    assert figure_name(first) != figure_name(second)
    assert "1130746797" in figure_name(first)
    assert figure_name(first).endswith(".png")


def test_figure_name_handles_empty_uid(tmp_path):
    entry = {"group": "Mode2_1609", "case_id": "DEV-A", "study_uid": "",
             "output": str(tmp_path / "full.marfi.bhc_denoised.raw")}
    assert figure_name(entry) == "Mode2_1609__DEV-A__uid__full.marfi.bhc.png"


def test_input_of_prefers_explicit_field(tmp_path):
    entry = {"output": str(tmp_path / "denoised" / "PDL" / "full_denoised.raw"),
             "input_path": str(tmp_path / "inputs" / "PDL" / "full.raw")}
    assert _input_of(entry) == tmp_path / "inputs" / "PDL" / "full.raw"


def test_input_of_derives_from_output_when_field_absent(tmp_path):
    entry = {"output": str(tmp_path / "denoised" / "PDL" / "Mode0_1615" / "full.bhc_denoised.raw")}
    derived = _input_of(entry)
    assert derived == tmp_path / "inputs" / "PDL" / "Mode0_1615" / "full.bhc.raw"


def test_input_of_ignores_non_path_input_key(tmp_path):
    """The ``input`` key holds statistics, so it must never be read as a path."""
    entry = {"output": str(tmp_path / "denoised" / "PDL" / "full_denoised.raw"),
             "input": {"sigma_immerkaer": 0.02}}
    assert _input_of(entry) == tmp_path / "inputs" / "PDL" / "full.raw"


def test_render_volume_case_writes_figure(tmp_path):
    shape = (8, 16, 16)
    inputs = tmp_path / "inputs" / "PDL" / "Mode2_1609" / "DEV-A" / "uid-a"
    denoised = tmp_path / "denoised" / "PDL" / "Mode2_1609" / "DEV-A" / "uid-a"
    inputs.mkdir(parents=True)
    denoised.mkdir(parents=True)
    source = np.random.default_rng(0).random(shape, dtype=np.float32).astype("<f4")
    source.tofile(inputs / "full.raw")
    (source * 0.5).astype("<f4").tofile(denoised / "full_denoised.raw")

    entry = {
        "group": "Mode2_1609", "case_id": "DEV-A", "study_uid": "uid-a",
        "domain": "full.raw", "shape": list(shape),
        "output": str(denoised / "full_denoised.raw"),
        "input_path": str(inputs / "full.raw"),
        "input": {"sigma_immerkaer": 0.02},
        "denoised": {"sigma_immerkaer": 0.01},
        "sigma_drop_pct": 50.0, "tissue_shift_rel_pct": 0.0,
    }

    written = render_volume_case(entry, tmp_path / "figures")
    assert written.exists() and written.stat().st_size > 0


def test_render_noise_summary_writes_figure(tmp_path):
    records = [
        {"group": "Mode0_1615", "case_id": "A", "domain": "full.raw",
         "input": {"sigma_immerkaer": 0.02}, "denoised": {"sigma_immerkaer": 0.01},
         "sigma_drop_pct": 50.0, "tissue_shift_rel_pct": 0.2},
        {"group": "Mode2_1609", "case_id": "B", "domain": "full.bhc.raw",
         "input": {"sigma_immerkaer": 0.03}, "denoised": {"sigma_immerkaer": 0.012},
         "sigma_drop_pct": 60.0, "tissue_shift_rel_pct": -0.1},
    ]
    written = render_noise_summary(records, tmp_path / "figures")
    assert written.exists() and written.stat().st_size > 0
