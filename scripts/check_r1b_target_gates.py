"""Gate the scheme-B recursive target chain before launching the next round.

Two overview JSONs can be produced by a round:

* ``prepare_full_<suffix>_overview.json`` -- the teacher denoising ``full``
  (every round).  Gate = the teacher really produced a *cleaner* target
  (Immerkaer delta > 0 and a relative reduction of at least
  ``--min-reduction``), otherwise the recursion degenerates.  For rounds
  >= 2 this is the ``full.dn_3d_128b_rN.raw`` teacher target and it is the
  **only** gate, because r2+ are plain recursion (target = teacher output,
  no blending).
* ``blend_full_<suffix>_overview.json`` -- the r1 mixed target (round 1 only).
  Because ``w`` clips to exactly 0 below ``lo`` and exactly 1 above ``hi``,
  the low band target must equal ``full`` and the high band target must equal
  the M0 output; the mid band must stay inside ``[full, M0]``.  Those are
  arithmetic identities of the mixer, so the tolerance only has to absorb the
  mean-of-means aggregation across studies -- a small tolerance therefore
  catches a wrong blend key or a swapped r1/r2 side.

Modes:

* ``--prep-overview`` alone -> prep-only gate (used for r2b/r3b/r4b).
* ``--prep-overview`` + ``--blend-overview`` -> stage A + stage B gate (r1b).

Exit code 0 = all gates pass (safe to start the round), 1 = stop and report.
Coverage is taken from the overview's ``rows`` list when present (the prepare
script merges rows across runs, so a resumed partial run still covers every
study); otherwise it falls back to ``n_new + n_existing``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: bands of the r1 target, ordered soft tissue -> bone
BANDS = (("low", "soft tissue"), ("mid", "crossfade"), ("high", "bone"))


def _rel(a: float, b: float) -> float:
    denom = abs(b) if abs(b) > 1e-12 else float("nan")
    return abs(a - b) / denom


def prep_coverage(prep: dict) -> int:
    """Number of studies covered by a prepare overview (rows preferred)."""
    rows = prep.get("rows")
    if isinstance(rows, list) and rows:
        return len(rows)
    return int(prep.get("n_new", 0)) + int(prep.get("n_existing", 0))


def evaluate_prep_gate(prep: dict, *, expect_studies: int = 200,
                       min_reduction: float = 0.05) -> list[str]:
    """Return the gate failures for the prepare (teacher target) overview."""
    failures: list[str] = []
    n_a = prep_coverage(prep)
    noise = prep.get("noise_immerkaer", {}) or {}
    delta = float(noise.get("delta_mean", float("nan")))
    red = float(noise.get("relative_reduction_mean", float("nan")))
    if int(prep.get("n_skip_missing", 0)) != 0:
        failures.append(f"stage A skipped {prep['n_skip_missing']} studies (missing full.raw)")
    if n_a != expect_studies:
        failures.append(f"stage A covered {n_a} studies, expected {expect_studies}")
    if not delta > 0:
        failures.append(f"stage A delta_mean={delta} is not > 0 (teacher did not clean full)")
    if not red >= min_reduction:
        failures.append(f"stage A relative reduction {red:.4f} < {min_reduction}")
    return failures


def prep_gate_warnings(prep: dict, *, max_reduction: float = 0.70) -> list[str]:
    """Non-fatal advisories on a prepare overview.

    A very large Immerkaer drop means the teacher flattened the volume as much
    as it denoised it -- the classic over-smoothing signature that plain
    recursion develops after a few rounds.  Immerkaer alone cannot separate
    "removed noise" from "removed texture", so this is a warning that asks for
    the banded QC, never a hard stop.
    """
    warnings: list[str] = []
    noise = prep.get("noise_immerkaer", {}) or {}
    red = float(noise.get("relative_reduction_mean", float("nan")))
    if red == red and max_reduction > 0 and red > max_reduction:
        warnings.append(f"relative reduction {red:.3f} > {max_reduction:.2f}: "
                        "possible over-smoothing -- run the banded QC before trusting this round")
    return warnings


def evaluate_blend_gate(blend: dict, *, expect_studies: int = 200,
                        band_tol: float = 0.02) -> list[str]:
    """Return the gate failures for the r1 blend (stage B) overview."""
    failures: list[str] = []
    n_b = int(blend.get("n_new", 0)) + int(blend.get("n_existing", 0))
    if int(blend.get("n_skip_missing", 0)) != 0:
        failures.append(f"stage B skipped {blend['n_skip_missing']} studies")
    if n_b != expect_studies:
        failures.append(f"stage B covered {n_b} studies, expected {expect_studies}")
    if blend.get("mask_source") != "high":
        failures.append("blend key must be the denoised M0 output "
                        f"(mask_source), got {blend.get('mask_source')!r}")

    bands = blend.get("bands", {}) or {}
    for band, _name in BANDS:
        row = bands.get(band)
        if not row:
            failures.append(f"blend overview lacks band {band}")
            continue
        full, r2, target = float(row["full"]), float(row["r2"]), float(row["target"])
        low_v, high_v = min(full, r2), max(full, r2)
        tol = band_tol * abs(high_v)
        if not (low_v - tol <= target <= high_v + tol):
            failures.append(f"band {band}: target {target:.5f} outside [full, M0] "
                            f"= [{low_v:.5f}, {high_v:.5f}]")
        if band == "low" and _rel(target, full) > band_tol:
            failures.append(f"low band target {target:.5f} != full {full:.5f} "
                            f"(rel {_rel(target, full):.4f} > {band_tol})")
        if band == "high":
            if _rel(target, r2) > band_tol:
                failures.append(f"high band target {target:.5f} != M0 {r2:.5f} "
                                f"(rel {_rel(target, r2):.4f} > {band_tol})")
            if not r2 < full:
                failures.append(f"high band M0 ({r2:.5f}) is not cleaner than full ({full:.5f})")
    return failures


def evaluate_gates(prep: dict, blend: dict, *, expect_studies: int = 200,
                   min_reduction: float = 0.05, band_tol: float = 0.02) -> list[str]:
    """Return the list of gate failures for the stage A + stage B overviews."""
    return (evaluate_prep_gate(prep, expect_studies=expect_studies,
                               min_reduction=min_reduction)
            + evaluate_blend_gate(blend, expect_studies=expect_studies, band_tol=band_tol))


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing overview: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a scheme-B recursion target before training.")
    parser.add_argument("--prep-overview", type=Path, required=True)
    parser.add_argument("--blend-overview", type=Path, default=None,
                        help="Stage-B blend overview. Omit for the plain-recursion rounds "
                             "(r2b/r3b/r4b), where the teacher target is the only artefact.")
    parser.add_argument("--expect-studies", type=int, default=200)
    parser.add_argument("--min-reduction", type=float, default=0.05,
                        help="Minimum mean relative Immerkaer reduction for the teacher target "
                             "(default 0.05).")
    parser.add_argument("--max-reduction", type=float, default=0.70,
                        help="Warn (never fail) above this mean relative reduction: "
                             "possible over-smoothing. 0 disables.")
    parser.add_argument("--band-tol", type=float, default=0.02,
                        help="Relative tolerance for the low/high band identity (default 0.02).")
    parser.add_argument("--label", default="next round",
                        help="Round label used in the final PASS/FAIL line (e.g. 'r2b').")
    args = parser.parse_args()

    prep = _load(args.prep_overview)
    blend = _load(args.blend_overview) if args.blend_overview is not None else None
    noise = prep.get("noise_immerkaer", {}) or {}

    print("=" * 72)
    print("STAGE A -- teacher target on full")
    n_a = prep_coverage(prep)
    print(f"  studies prepared : {n_a} (rows={len(prep.get('rows', []) or [])} "
          f"new={prep.get('n_new')} existing={prep.get('n_existing')} "
          f"skipped={prep.get('n_skip_missing')})")
    print(f"  immerkaer full   : {noise.get('full_mean')}")
    print(f"  immerkaer dn     : {noise.get('dn_mean')}")
    print(f"  delta mean       : {noise.get('delta_mean')}")
    red = float(noise.get("relative_reduction_mean", float("nan")))
    print(f"  relative drop    : {red * 100:.2f}%  (n={noise.get('n')})"
          if red == red else "  relative drop    : nan")

    if blend is not None:
        print("-" * 72)
        print("STAGE B -- r1 mixed target (w=0 in the low band, w=1 in the high band)")
        n_b = int(blend.get("n_new", 0)) + int(blend.get("n_existing", 0))
        print(f"  lo={blend.get('lo')} hi={blend.get('hi')} kind={blend.get('kind')} "
              f"mask_source={blend.get('mask_source')} splits={blend.get('splits')}")
        print(f"  studies blended  : {n_b} (new={blend.get('n_new')} "
              f"existing={blend.get('n_existing')} skipped={blend.get('n_skip_missing')})")
        for band, name in BANDS:
            row = (blend.get("bands") or {}).get(band)
            if not row:
                print(f"  [{band:<4}] {name:<11} MISSING")
                continue
            print(f"  [{band:<4}] {name:<11} full={float(row['full']):.5f} "
                  f"M0={float(row['r2']):.5f} target={float(row['target']):.5f} (n={row.get('n')})")
    else:
        print("-" * 72)
        print("STAGE B -- skipped (plain-recursion round: target = teacher output)")

    failures = evaluate_prep_gate(prep, expect_studies=args.expect_studies,
                                  min_reduction=args.min_reduction)
    if blend is not None:
        failures += evaluate_blend_gate(blend, expect_studies=args.expect_studies,
                                        band_tol=args.band_tol)
    warnings = prep_gate_warnings(prep, max_reduction=args.max_reduction)

    print("=" * 72)
    for line in warnings:
        print(f"WARN: {line}")
    if failures:
        print(f"GATE RESULT: FAIL ({len(failures)} problem(s))")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"GATE RESULT: PASS -- safe to start {args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
