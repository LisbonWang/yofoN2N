"""Copy the full.raw reference volume into each case dir of an eval output dir
(keeps dir structure identical to E1/E2/E3 dirs: sources + ref + denoised)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Manifest containing full.raw paths for the selected split.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.parent
    records = {str(r["key"]): r for r in manifest["records"]}
    split_keys = list(manifest["splits"][args.split])

    n = 0
    for key in split_keys:
        rec = records[key]
        group = rec.get("group") or "?"
        uid = rec.get("study_uid") or rec.get("key")
        case_dir = args.output_dir / f"{group}__{uid}"
        rel = (rec.get("outputs") or {}).get("full.raw")
        if rel is None:
            print(f"[skip] {key}: no full.raw in outputs", flush=True)
            continue
        src = base / rel
        dst = case_dir / "full.raw"
        if not src.exists():
            print(f"[skip] {key}: source missing {src}", flush=True)
            continue
        if dst.exists():
            print(f"[exists] {dst}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        n += 1
        print(f"[copied] {dst}", flush=True)
    print(f"done: {n} reference volumes copied to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
