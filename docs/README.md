# Documentation Guide

## Reusable workflow

- [Project setup, architecture, training, inference, and remote execution](../README.md)
- [RAW data and manifest contract](data-format.md)
- [Portable and experiment-specific config inventory](../configs/README.md)

## Experiment records

Documents under `plans/`, `reports/`, and `superpowers/` may preserve dataset identifiers, machine paths, measured results, or one-off procedures so that the associated work can be traced. Treat them as historical records unless a document explicitly says it is a current reusable guide. Do not copy their private paths into new configs or publish manifests/volumes that identify people or studies.

- [1609 training plan](training_plan_1609.md): AIReduceNoise historical dataset and training snapshot.
- [Recursive blend plan](plans/recursive-blend-at-r1.md): AIR3 distillation design plus archived earlier plan.
- [JIROX-II reconstruction report](reports/jirox2-recon-status.md): dated status for one private-data batch.
- [AIR3 source-domain report](reports/air3-source-domain-gap.md): experiment-specific volume observations.

`configs/recon_filters/` and `configs/recon_groups.json` also contain scanner/site-specific XDay geometry and filter settings. Validate them against the local acquisition protocol and SDK before use.

Scripts named for AIR3 rounds, `redenoise_case_fp16.py`, `eval_rounds_banded.py`, `make_denoise_figures.py`, `_review_*`, `_check_*`, and `_validate_*` helpers are experiment utilities. Their fixed manifests, paths, report names, or assumptions belong to archived runs; use the explicit-argument scripts and configs for new workflows. Benchmark scripts may also contain run-specific device and shape settings; treat those as measurement records rather than hardware requirements.
