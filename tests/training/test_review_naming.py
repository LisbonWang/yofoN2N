"""Tests for the expert-review package naming scheme.

These pin the two things that actually broke during the rename migration:

* ``volume_<X>X<Y>X<Z>_f32`` must carry the ``.raw`` extension, and the
  ``<X>X<Y>X<Z>`` order is fdkPara (x, y, z) -- so the *slice count comes last*
  even though the array shape is (z, y, x);
* the legacy -> new replacement pairs must be applied longest-name-first, or
  ``...rcoor.raw`` eats the prefix of ``...rcoor.raw.json`` and
  ``...ai.r4b.raw`` eats the prefix of ``...ai.r4b.gfsharp_s0.3.raw``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts._review_assemble import (
    BYTES_TO_SHAPE,
    NAME_SUFFIXES,
    SHARPEN_ARTEFACTS,
    SHARPENED_INPUTS,
    dims_tag,
    vol_names,
)
from scripts._review_rename import EXT_REPAIR, LEGACY, rename_rules

VOL600 = (600, 640, 640)
VOL360 = (360, 640, 640)


@pytest.mark.parametrize("shape,expected", [
    (VOL600, "640X640X600"),
    (VOL360, "640X640X360"),
])
def test_dims_tag_is_xyz_not_zyx(shape, expected):
    # shape is (z, y, x) but the name follows fdkPara (x, y, z).
    assert dims_tag(shape) == expected


def test_dims_tag_reverses_the_array_shape():
    assert dims_tag((2, 3, 4)) == "4X3X2"


@pytest.mark.parametrize("shape,dims", [(VOL600, "640X640X600"), (VOL360, "640X640X360")])
def test_vol_names_are_flat_with_one_extension(shape, dims):
    names = [n for n, _, _ in vol_names(shape)]
    assert names == [
        f"volume_{dims}_f32.raw",
        f"volume_{dims}_f32_traditional.raw",
        f"volume_{dims}_f32_ai_r1b.raw",
        f"volume_{dims}_f32_ai_r2b.raw",
        f"volume_{dims}_f32_ai_r3b.raw",
        f"volume_{dims}_f32_ai_r4b.raw",
        f"volume_{dims}_f32_ai_r4b_sharp.raw",
        f"volume_{dims}_f32_ai_r3b_sharp.raw",
    ]


def test_vol_names_never_lose_the_raw_extension():
    # The first migration pass wrote the stem without ".raw"; guard it explicitly.
    for shape in (VOL600, VOL360):
        for name, _, _ in vol_names(shape):
            assert name.endswith(".raw")
            assert not name.endswith(".raw.raw")
            # exactly one dot: "volume_..._f32_suffix.raw"
            assert name.count(".") == 1


def test_vol_names_have_no_dot_before_the_ai_tag():
    for name, key, _ in vol_names(VOL360):
        assert ".." not in name
        assert ".ai." not in name
        if key.startswith("ai_"):
            assert f"_{key}." in name


def test_vol_names_keys_match_the_suffix_table():
    assert [k for _, k, _ in vol_names(VOL600)] == [k for _, k, _ in NAME_SUFFIXES]
    assert len(vol_names(VOL600)) == 8


def test_every_sharpened_volume_has_a_plain_source_to_sharpen():
    """_ai_r{3,4}b_sharp must each point at a volume that also exists."""
    keys = {k for _, k, _ in vol_names(VOL600)}
    for out_key, suffix in SHARPENED_INPUTS.items():
        assert out_key in keys, out_key
        assert "sharp" not in suffix, suffix
        assert f"{suffix[1:]}" in keys, suffix          # "_ai_r4b" -> "ai_r4b"
    assert set(SHARPEN_ARTEFACTS) == set(SHARPENED_INPUTS)


def test_bytes_to_shape_covers_both_geometries():
    assert BYTES_TO_SHAPE[600 * 640 * 640 * 4] == VOL600
    assert BYTES_TO_SHAPE[360 * 640 * 640 * 4] == VOL360


def test_bytes_to_shape_ignores_a_json_sidecar_size():
    # A sidecar is a few kB, never a whole volume; the migration relies on this to
    # infer geometry from byte size alone.
    assert BYTES_TO_SHAPE.get(2500) is None


# ------------------------------------------------------------------ rename rules

def test_rename_rules_are_longest_first():
    rules = rename_rules("640X640X360")
    olds = [old for old, _ in rules]
    assert olds == sorted(olds, key=len, reverse=True)


def test_rename_rules_never_leave_a_bare_stem():
    for old, new in rename_rules("640X640X360"):
        if old.endswith(".raw"):
            assert new.endswith(".raw"), (old, new)
        assert not new.endswith(".raw.raw")


def test_rename_rules_cover_every_volume_plus_both_sidecars():
    rules = dict(rename_rules("640X640X600"))
    assert len(rules) == 10                                  # 8 volumes + 2 sidecars
    assert rules["input.bhc.marfi.rcoor.raw"] == "volume_640X640X600_f32.raw"
    assert rules["input.bhc.marfi.rcoor.raw.json"] == "volume_640X640X600_f32.raw.json"
    assert rules["input.bhc.marfi.rcoor.conv.rn_ani.raw"] == \
        "volume_640X640X600_f32_traditional.raw"
    assert rules["input.bhc.marfi.rcoor.conv.rn_ani.raw.json"] == \
        "volume_640X640X600_f32_traditional.raw.json"
    assert rules["input.bhc.marfi.rcoor.ai.r4b.gfsharp_s0.3.raw"] == \
        "volume_640X640X600_f32_ai_r4b_sharp.raw"
    assert rules["input.bhc.marfi.rcoor.ai.r3b.gfsharp_s0.3.raw"] == \
        "volume_640X640X600_f32_ai_r3b_sharp.raw"


def test_rename_rules_apply_without_prefix_corruption():
    """The real failure mode: sequential plain replaces on a JSON blob."""
    blob = ('{"a": "input.bhc.marfi.rcoor.raw", '
            '"b": "input.bhc.marfi.rcoor.raw.json", '
            '"c": "input.bhc.marfi.rcoor.ai.r4b.gfsharp_s0.3.raw", '
            '"d": "input.bhc.marfi.rcoor.ai.r4b.raw", '
            '"e": "input.bhc.marfi.rcoor.ai.r3b.gfsharp_s0.3.raw"}')
    for old, new in rename_rules("640X640X360"):
        blob = blob.replace(old, new)
    assert '"a": "volume_640X640X360_f32.raw"' in blob
    assert '"b": "volume_640X640X360_f32.raw.json"' in blob
    assert '"c": "volume_640X640X360_f32_ai_r4b_sharp.raw"' in blob
    assert '"d": "volume_640X640X360_f32_ai_r4b.raw"' in blob
    assert '"e": "volume_640X640X360_f32_ai_r3b_sharp.raw"' in blob
    # no double extensions, no leftovers, no mixed-up tags
    assert ".raw.raw" not in blob
    assert "input.bhc" not in blob
    assert "gfsharp" not in blob
    assert "_sharp.raw.json" not in blob


def test_rename_rules_are_mutually_consistent_with_vol_names():
    """Every new name a rule produces must be one the assembler expects."""
    expected = {name for name, _, _ in vol_names(VOL360)}
    produced = {new for old, new in rename_rules("640X640X360") if old.endswith(".raw")}
    assert produced == expected


def test_legacy_table_covers_every_suffix_of_the_new_scheme():
    known = {s for s, _, _ in NAME_SUFFIXES}
    assert set(LEGACY.values()) == known


# ------------------------------------------------------- extension repair regexes

@pytest.mark.parametrize("text,expected", [
    ("volume_640X640X360_f32", "volume_640X640X360_f32.raw"),
    ("volume_640X640X360_f32_ai_r1b", "volume_640X640X360_f32_ai_r1b.raw"),
    ("volume_640X640X360_f32_ai_r4b_sharp", "volume_640X640X360_f32_ai_r4b_sharp.raw"),
    ("volume_640X640X360_f32_traditional", "volume_640X640X360_f32_traditional.raw"),
])
def test_ext_repair_adds_the_missing_extension(text, expected):
    for pat, repl in EXT_REPAIR:
        text = pat.sub(repl, text)
    assert text == expected


@pytest.mark.parametrize("text", [
    "volume_640X640X360_f32.raw",
    "volume_640X640X360_f32.raw.json",
    "volume_640X640X360_f32_ai_r1b.raw",
    "volume_640X640X360_f32_traditional.raw.json",
])
def test_ext_repair_leaves_correct_names_alone(text):
    out = text
    for pat, repl in EXT_REPAIR:
        out = pat.sub(repl, out)
    assert out == text


def test_ext_repair_is_idempotent():
    text = "volume_640X640X600_f32 and volume_640X640X600_f32_ai_r2b"
    once = text
    for pat, repl in EXT_REPAIR:
        once = pat.sub(repl, once)
    twice = once
    for pat, repl in EXT_REPAIR:
        twice = pat.sub(repl, twice)
    assert once == twice


def test_ext_repair_does_not_touch_unrelated_metric_keys():
    # metrics.json/case.json carry volume_min / volume_std -- they must survive.
    text = '{"volume_min": -0.6, "volume_max": 2.8, "volume_std": 0.12}'
    out = text
    for pat, repl in EXT_REPAIR:
        out = pat.sub(repl, out)
    assert out == text


def test_ext_repair_handles_absolute_windows_paths():
    text = r"G:\workspace\N2N\outputs\expert_review_ai_vs_conv\c01_x\volume_640X640X600_f32_ai_r4b"
    for pat, repl in EXT_REPAIR:
        text = pat.sub(repl, text)
    assert text.endswith(r"volume_640X640X600_f32_ai_r4b.raw")


def test_is_bare_volume_recognises_only_extensionless_volume_files():
    from scripts._review_rename import is_bare_volume
    assert is_bare_volume(Path("volume_640X640X360_f32_ai_r1b"))
    assert not is_bare_volume(Path("volume_640X640X360_f32_ai_r1b.raw"))
    assert not is_bare_volume(Path("volume_640X640X360_f32.raw.json"))
    assert not is_bare_volume(Path("case.json"))
    assert not is_bare_volume(Path("metrics_ai.json"))
