"""Per-group analysis of the AIR3 128b banded validation (plan B, r0 -> r4b).

Reads ``outputs/eval_air3_128b_banded_test11.json`` and prints, per acquisition
group and overall:

* banded Immerkaer sigma per checkpoint (full, M0, M1b..M4b);
* reduction vs full, both total and marginal (round-over-round);
* ``high / low`` sigma ratio -- the *band differentiation* scalar.  Plan B's
  whole premise is that the r1 blend makes bone one recursion deeper than soft
  tissue; if that ratio drifts back to ~1.0 the differentiation has washed out.

Read-only; nothing is written.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "outputs" / "eval_air3_128b_banded_test11.json"

MODELS = ("full", "M0", "M1b", "M2b", "M3b", "M4b")
ROUNDS = ("M0", "M1b", "M2b", "M3b", "M4b")


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def table(rows: list[dict], title: str) -> None:
    acc = {b: {m: [] for m in MODELS} for b in ("low", "mid", "high")}
    gacc = {m: [] for m in MODELS}
    for row in rows:
        for band, entry in row["bands"].items():
            for m in MODELS:
                if entry.get(m) is not None:
                    acc[band][m].append(entry[m])
        for m in MODELS:
            if row["global"].get(m) is not None:
                gacc[m].append(row["global"][m])

    print(f"\n########## {title} (n={len(rows)}) ##########")
    hdr = f"{'band':>6} | " + " | ".join(f"{m:>7}" for m in MODELS)
    print(hdr)
    print("-" * len(hdr))
    for band in ("low", "mid", "high"):
        print(f"{band:>6} | " + " | ".join(f"{mean(acc[band][m]):7.5f}" for m in MODELS))
    print(f"{'GLOBAL':>6} | " + " | ".join(f"{mean(gacc[m]):7.5f}" for m in MODELS))

    print(f"\n  -- reduction vs full (%) --")
    for band in ("low", "mid", "high"):
        base = mean(acc[band]["full"])
        print(f"{band:>6} | " + " | ".join(
            f"{100 * (1 - mean(acc[band][m]) / base):7.1f}" for m in MODELS))
    gbase = mean(gacc["full"])
    print(f"{'GLOBAL':>6} | " + " | ".join(f"{100 * (1 - mean(gacc[m]) / gbase):7.1f}" for m in MODELS))

    print(f"\n  -- marginal reduction round-over-round (%) --")
    for band in ("low", "mid", "high"):
        vals = [mean(acc[band][m]) for m in ROUNDS]
        marg = [f"{100 * (1 - vals[i] / vals[i - 1]):7.1f}" for i in range(1, len(vals))]
        labels = [f"{ROUNDS[i - 1]}->{ROUNDS[i]}" for i in range(1, len(ROUNDS))]
        print(f"{band:>6} | " + "  ".join(f"{lab}={v}" for lab, v in zip(labels, marg, strict=True)))
    gvals = [mean(gacc[m]) for m in ROUNDS]
    print(f"{'GLOBAL':>6} | " + "  ".join(
        f"{ROUNDS[i - 1]}->{ROUNDS[i]}={100 * (1 - gvals[i] / gvals[i - 1]):.1f}"
        for i in range(1, len(ROUNDS))))

    print(f"\n  -- BAND DIFFERENTIATION: sigma(high) / sigma(low) --")
    print(f"{'':>6} | " + " | ".join(f"{m:>7}" for m in MODELS) + "   (1.00 = no density adaptation)")
    ratios = []
    for m in MODELS:
        ratio = mean(acc["high"][m]) / mean(acc["low"][m])
        ratios.append((m, ratio))
    print(f"{'h/l':>6} | " + " | ".join(f"{r:7.3f}" for _, r in ratios))
    # per-case spread for the key rounds
    for m in ("full", "M1b", "M2b", "M4b"):
        per = [r["bands"]["high"][m] / r["bands"]["low"][m] for r in rows
               if r["bands"]["low"].get(m) and r["bands"]["high"].get(m)]
        print(f"   {m:>4} per-case h/l: " + " ".join(f"{v:.3f}" for v in per)
              + f"   [min {min(per):.3f} max {max(per):.3f}]")


def main() -> None:
    data = json.loads(JSON.read_text(encoding="utf-8"))
    per_case = data["per_case"]
    print(f"manifest : {data['manifest']}")
    print(f"split    : {data['split']}   studies: {data['n_studies']}   band_key: {data['band_key']}")
    print(f"bands    : lo={data['lo']} hi={data['hi']} tissue_floor={data['tissue_floor']}")

    table(per_case, "ALL 11 (cross-mode)")
    groups: dict[str, list[dict]] = {}
    for row in per_case:
        groups.setdefault(str(row.get("group")), []).append(row)
    for g in sorted(groups):
        table(groups[g], g)

    mode2 = [r for r in per_case if r.get("group") == "Mode2_1609"]
    if mode2:
        table(mode2, "Mode2_1609 only (in-domain)")
    other = [r for r in per_case if r.get("group") != "Mode2_1609"]
    if other:
        table(other, "Mode0/1/3 (off-domain)")


if __name__ == "__main__":
    main()
