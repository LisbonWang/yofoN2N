# Mode2 Reconstruction Filter

## Goal

Process only Mode2 cases from the source scan directory before XDay reconstruction. The observed source contains 47 Mode2 cases and 3 Mode1 cases.

## Design

- Add an optional mode filter to discover_cases and the prepare_pairs.py CLI.
- When supplied, discovery returns only cases whose parsed mode matches case-insensitively.
- --mode Mode2 is explicit and recommended for the provided source. Omitting it preserves all-mode behavior.
- If a requested mode has no matches, fail clearly rather than silently producing an empty dataset.
- Filtering occurs before dry-run output and reconstruction, so excluded cases incur no SDK work or output artifacts.
- Pair generation, projection subsetting, manifests, and training remain unchanged.

## Verification

- Unit tests cover case filtering and no-match errors.
- CLI help exposes --mode.
- The provided Z-drive dry-run with --mode Mode2 reports 47 cases and performs no reconstruction.
