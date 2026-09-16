"""Orchestrate one recursive training round (half -> full_denoised).

The recursive scheme turns a converged ``half -> full`` model into a stronger
one by iterating on the *supervision target*:

    round 0:  M0 = train(even/odd -> full)                        # plain half->full
    round k:  teacher = M_{k-1}
              full.dn{k}.raw = M_{k-1}(full)                       # teacher denoises full
              M_k = train(even/odd -> full.dn{k}, init-weights=M_{k-1})

This script automates one round (``--round k``) end-to-end:

  1. runs :mod:`prepare_full_dn_targets` to produce ``full.dn{k}.raw`` for the
     train (+val) splits and register them in the manifest,
  2. derives a student config whose mappings are
     ``[even->full.dn{k}, odd->full.dn{k}]`` (everything else identical to the
     baseline half->full config -- control variable is only the target),
  3. prints the exact training command, warm-started from the teacher via
     ``--init-weights`` (NOT ``--resume``: cold optimizer/scheduler/LR, epoch 0).

Run the printed command on the server (tmux), then -- if the round's
``full.dn{k} vs full`` noise-level diagnostic shows the target is actually
cleaner (lower noise std) -- recurse with ``--round k+1 --teacher <M_k best.pt>``.

Leakage rule: only ``train``/``val`` get teacher-denoised targets.  The
``test`` split is never touched by the teacher and is only used for the final
evaluation.  Because ``full`` is itself noisy, the correct success metric is
the output's noise level (lower is better), NOT PSNR against ``full``; PSNR is
reported for reference only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parent
_PREPARE = _SCRIPTS / "prepare_full_dn_targets.py"

#: arch -> training entry point (3D uses train_voxel; 2D/2D-DS use train_sparse_dense)
_ENTRY_BY_ARCH = {
    "unet3d": "scripts/train_voxel.py",
    "unet2d": "scripts/train_sparse_dense.py",
    "unet2d_ds": "scripts/train_sparse_dense.py",
}


def _round_suffix(round_index: int) -> str:
    """Target suffix for a round: round 1 -> ``dn1``, round 2 -> ``dn2`` ..."""
    return f"dn{int(round_index)}"


def _derive_config(config_path: Path, suffix: str, round_index: int) -> Path:
    """Write a student config identical to the baseline except for the target."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["mappings"] = [f"even->full.{suffix}", f"odd->full.{suffix}"]
    # 3D recursive training: stride == voxel_depth (non-overlapping windows) so
    # each training target window is exactly one complete teacher output block
    # (the teacher denoises full with a non-overlapping stride=voxel_depth
    # sliding window).  Overlapping windows would straddle two teacher output
    # blocks and mix two forward passes into a single label.
    data = config.setdefault("data", {})
    if "voxel_depth" in data:
        data["stride"] = int(data.get("voxel_depth", 16))
    base_run_dir = str(config["training"].get("run_dir", "runs/recursive"))
    config["training"]["run_dir"] = f"{base_run_dir}_{suffix}"
    derived = config_path.with_name(f"{config_path.stem}_{suffix}.yaml")
    derived.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate one recursive round (half -> full_denoised).")
    parser.add_argument("--config", type=Path, required=True,
                        help="Baseline half->full config (e.g. configs/1609v2_unet2d_ds_h2f.yaml).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True,
                        help="Converged teacher checkpoint (previous round's best.pt).")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--target-suffix", default=None,
                        help="Override the target suffix (default 'dn{round}'). Use a distinct "
                             "suffix (e.g. 'dn3d') to separate teachers of different "
                             "architectures so their full.<suffix>.raw files never collide.")
    parser.add_argument("--teacher-arch", choices=("unet2d", "unet2d_ds", "unet3d"), default=None,
                        help="Teacher arch for inference (default: read from --config).")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0",
                        help="Device for the teacher inference pass (target generation).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the derived config + command without generating targets.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    suffix = args.target_suffix or _round_suffix(args.round)
    arch = args.teacher_arch or str(config.get("model", {}).get("arch", "unet2d"))
    entry = _ENTRY_BY_ARCH[arch]
    base_run_dir = str(config["training"].get("run_dir", "runs/recursive"))
    student_run_dir = f"{base_run_dir}_{suffix}"

    print(f"[recursive] round={args.round} suffix={suffix} teacher={args.teacher} "
          f"arch={arch} entry={entry} run_dir={student_run_dir}", flush=True)

    if not args.dry_run:
        cmd = [
            sys.executable, str(_PREPARE),
            "--config", str(args.config),
            "--manifest", str(args.manifest),
            "--checkpoint", str(args.teacher),
            "--arch", arch,
            "--target-suffix", suffix,
            "--splits", args.splits,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ]
        print("[recursive] generating targets: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    derived = _derive_config(args.config, suffix, args.round)
    print(f"\n[recursive] student config written: {derived}", flush=True)

    train_cmd = (
        f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/anaconda3/envs/pytorch201/bin/python "
        f"{entry} --arch {arch} --config {derived} --manifest {args.manifest} "
        f"--run-dir {student_run_dir} "
        f"--init-weights {args.teacher}"
    )
    print("\n=== training command (run inside tmux on the server) ===")
    print(train_cmd)
    print("\n=== after it finishes ===")
    print(f"  evaluate:  python scripts/test_raw_3d.py --config {derived} "
          f"--manifest {args.manifest} --checkpoint {student_run_dir}/best.pt "
          f"--split test --input-kind full --save-volumes raw")
    print(f"  gate:      check prepare_full_{suffix}_overview.json -> noise_immerkaer "
          f"'delta' > 0 (target is lower-noise than full), NOT PSNR.")
    print(f"  next round: python scripts/run_recursive_round.py --config {derived} "
          f"--manifest {args.manifest} --teacher {student_run_dir}/best.pt "
          f"--round {args.round + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
