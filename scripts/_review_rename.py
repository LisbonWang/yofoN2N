"""Flatten an already-built expert-review package to the new volume naming.

The first cut of the package used the inference/recon internals as file names::

    input.bhc.marfi.rcoor.ai.r4b.gfsharp_s0.3.raw

which is accurate but unreadable in a delivery folder.  The agreed scheme is one
flat, single-extension name::

    volume_<X>X<Y>X<Z>_f32[_后缀].raw

with ``X/Y/Z`` taken from ``fdkPara`` (``xNum/yNum/zNum``, so the slice count comes
last): ``640X640X600`` for the 1615 groups, ``640X640X360`` for the 1609 ones.

This is an *in-place migration*: it renames the volumes and rewrites every JSON
that points at them, so no 43 GiB is rewritten.  Byte sizes are checked before and
after.  Root-level ``manifest.json`` / ``metrics.json`` / ``README.md`` are left
alone on purpose -- ``_review_assemble.py`` regenerates them from scratch, which is
the only correct way to rebuild them (the same old basename maps to two different
new names depending on the group, so a blind text substitution would put the wrong
geometry into half the rows).

Text is patched longest-old-name-first.  That ordering is what keeps
``...rcoor.raw`` from eating the prefix of ``...rcoor.raw.json`` and
``...ai.r4b.raw`` from eating the prefix of ``...ai.r4b.gfsharp_s0.3.raw``.

The script also *normalises*: any ``volume_*`` file sitting there without its
``.raw`` extension gets it back, and path references inside JSON/markdown get the
same treatment.  That repairs a first pass that wrote the stem without the
extension, and it is idempotent, so the script can be re-run safely at any time.

Usage::

    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/_review_rename.py --dry-run
    C:/Users/admin/anaconda3/envs/torch/python.exe scripts/_review_rename.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._review_assemble import BYTES_TO_SHAPE, NAME_SUFFIXES, dims_tag

# The naming the first build used, keyed by the suffix the new scheme maps it to.
# Keys carry no leading dot: they are joined to LEGACY_STEM with one.
LEGACY_STEM = "input.bhc.marfi.rcoor"
LEGACY = {
    "raw": "",
    "conv.rn_ani.raw": "_traditional",
    "ai.r1b.raw": "_ai_r1b",
    "ai.r2b.raw": "_ai_r2b",
    "ai.r3b.raw": "_ai_r3b",
    "ai.r4b.raw": "_ai_r4b",
    "ai.r4b.gfsharp_s0.3.raw": "_ai_r4b_sharp",
}

# Text-only rules for identifiers that are not file paths: the model keys that
# denoise_multi_models.py embeds in metrics_ai.json.  Quoted so they cannot touch
# the same token inside a file name.
MODEL_TOKEN_RULES = [("ai.r1b", "ai_r1b"), ("ai.r2b", "ai_r2b"),
                     ("ai.r3b", "ai_r3b"), ("ai.r4b", "ai_r4b")]

TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv"}


def suffix_lookup() -> dict[str, str]:
    """new-suffix -> new-suffix, sanity check that LEGACY covers the scheme."""
    known = {s for s, _, _ in NAME_SUFFIXES}
    missing = set(LEGACY.values()) - known
    if missing:
        raise SystemExit(f"LEGACY maps to suffixes the new scheme does not define: "
                         f"{sorted(missing)}")
    return {s: s for s in known}


def rename_rules(dims: str) -> list[tuple[str, str]]:
    """(old, new) basename pairs for one case, longest old name first."""
    base = f"volume_{dims}_f32"
    rules = {f"{LEGACY_STEM}.{old}": f"{base}{new}.raw" for old, new in LEGACY.items()}
    # sidecars written by recon_conventional_denoise.py for the two recon chains
    rules[f"{LEGACY_STEM}.raw.json"] = f"{base}.raw.json"
    rules[f"{LEGACY_STEM}.conv.rn_ani.raw.json"] = f"{base}_traditional.raw.json"
    return sorted(rules.items(), key=lambda kv: len(kv[0]), reverse=True)


# The first migration pass forgot the ".raw" extension, leaving bare volume stems
# behind ("volume_640X640X600_f32_ai_r1b").  A plain replace cannot repair that:
# the bare-stem rule would also match the prefix of an already-correct name, so the
# repair has to be anchored.  These patterns are idempotent -- the lookaheads skip
# anything that already carries the extension.
EXT_REPAIR = [
    # <base>_ai_r1b / ..._ai_r4b / ..._ai_r4b_sharp / ..._traditional
    (re.compile(r"(volume_\d+X\d+X\d+_f32_(?:ai_r[1-4]b(?:_sharp)?|traditional))"
                r"(?!\.raw)"), r"\1.raw"),
    # bare <base>: not followed by "_" (a suffix) or "." (sidecar / extension)
    (re.compile(r"(volume_\d+X\d+X\d+_f32)(?![._])"), r"\1.raw"),
]


def is_bare_volume(path: Path) -> bool:
    """A ``volume_*`` file that lost its ``.raw`` extension."""
    return path.name.startswith("volume_") and path.suffix == ""


def infer_shape_any(directory: Path) -> tuple[int, int, int]:
    """Geometry from any volume-sized file in the folder.

    Deliberately more forgiving than ``_review_assemble.infer_case_shape``, which
    globs ``*.raw``: the migration must also cope with volumes whose extension is
    missing, so it matches on byte size alone and skips the JSON sidecars (their
    sizes never coincide with a volume).
    """
    shapes = set()
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        shape = BYTES_TO_SHAPE.get(f.stat().st_size)
        if shape is not None:
            shapes.add(shape)
    if not shapes:
        raise SystemExit(f"{directory.name}: no volume-sized file; cannot infer geometry")
    if len(shapes) > 1:
        raise SystemExit(f"{directory.name}: mixed geometry {sorted(shapes)}")
    return shapes.pop()


def patch_text(path: Path, rules: list[tuple[str, str]]) -> int:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in rules:                      # already longest-first
        if old in text:
            text = text.replace(old, new)
    for pat, repl in EXT_REPAIR:
        text = pat.sub(repl, text)
    for old, new in MODEL_TOKEN_RULES:
        text = text.replace(f'"{old}"', f'"{new}"')
    if text == original:
        return 0
    path.write_text(text, encoding="utf-8")
    return sum(1 for a, b in zip(original.splitlines(), text.splitlines()) if a != b) or 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "outputs" / "expert_review_ai_vs_conv")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan without touching anything.")
    args = p.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"package root not found: {root}")

    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("c"))
    if not dirs:
        raise SystemExit(f"no case directories under {root}")

    log_dir = root / "_logs"
    record: dict = {"root": str(root), "cases": {}}
    n_renamed = n_patched = n_already = n_repaired = 0

    for d in dirs:
        tag = d.name.split("_")[0]                      # "c01"
        shape = infer_shape_any(d)
        dims = dims_tag(shape)
        rules = rename_rules(dims)
        case_rec = {"dir": d.name, "dims": dims, "renamed": {}, "already": [],
                    "repaired": [], "patched": []}

        for old, new in rules:
            src, dst = d / old, d / new
            if src.is_file():
                size = src.stat().st_size
                if dst.exists():
                    raise SystemExit(f"{d.name}: refusing to overwrite {new}")
                if args.dry_run:
                    print(f"[plan ] {tag} {old}  ->  {new}")
                else:
                    src.rename(dst)
                    if dst.stat().st_size != size:
                        raise SystemExit(f"{d.name}: size changed renaming {old}")
                case_rec["renamed"][old] = new
                n_renamed += 1
            elif dst.is_file():
                case_rec["already"].append(new)
                n_already += 1

        # Normalise the extension: the first migration pass wrote bare stems.
        for f in sorted(d.iterdir()):
            if not f.is_file() or not is_bare_volume(f):
                continue
            if args.dry_run:
                print(f"[plan ] {tag} {f.name}  ->  {f.name}.raw   (extension repair)")
            else:
                f.rename(f.with_name(f.name + ".raw"))
            case_rec["repaired"].append(f"{f.name}.raw")
            n_repaired += 1

        # Text that points at the volumes: this case's own JSON/markdown, plus the
        # sharpening artefacts parked in _logs (the case folders stay raw-only).
        text_files = [f for f in sorted(d.iterdir())
                      if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES]
        text_files += [f for f in sorted((log_dir / "gfsharp" / tag).rglob("*"))
                       if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES]
        for f in text_files:
            if args.dry_run:
                case_rec["patched"].append(str(f.relative_to(root)))
                continue
            if patch_text(f, rules):
                case_rec["patched"].append(str(f.relative_to(root)))
                n_patched += 1

        print(f"[{tag}] dims={dims}  renamed={len(case_rec['renamed'])}  "
              f"already={len(case_rec['already'])}  repaired={len(case_rec['repaired'])}  "
              f"patched={len(case_rec['patched'])}")
        record["cases"][tag] = case_rec

    print(f"\nrenamed {n_renamed} file(s), repaired {n_repaired} extension(s), "
          f"patched {n_patched} text file(s), {n_already} already in the new scheme")
    if args.dry_run:
        print("(dry run -- nothing written)")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "rename_map.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[map] {out}")
    print("\nNext: re-run scripts/_review_assemble.py so case.json / manifest.json / "
          "metrics.json / README.md are regenerated against the new names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
