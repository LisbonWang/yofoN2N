# Mode2 Reconstruction Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Filter cases by acquisition mode before XDay reconstruction so the provided workflow processes only 47 Mode2 cases.

**Architecture:** Extend `discover_cases` with an optional case-insensitive mode predicate. The prepare CLI passes `--mode` into discovery before printing, dry-run handling, and reconstruction; no pair, model, or manifest behavior changes.

**Tech Stack:** Python 3.11, pytest, argparse, pathlib.

## Global Constraints

- Source `Z:\raid5\共享数据\典型图像\长安管口-PDL` remains read-only.
- `--mode Mode2` must exclude all Mode1 cases before SDK invocation.
- Omitting `--mode` preserves all-mode discovery.
- An explicitly requested mode with no matches must fail clearly.

---

### Task 1: Discovery Mode Predicate

**Files:**
- Modify: `src/cbct_n2n/reconstruction/discovery.py`
- Test: `tests/reconstruction/test_discovery.py`

**Interfaces:**
- `discover_cases(root, *, continue_on_error=False, mode=None) -> list[ScanCase]`

- [ ] **Step 1: Write failing tests**

Add a fixture with one Mode1 and one Mode2 raw/config pair; assert `discover_cases(tmp_path, mode="Mode2")` returns only Mode2. Add a no-match assertion expecting `ValueError` containing the requested mode.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest -q tests/reconstruction/test_discovery.py`
Expected: FAIL because `discover_cases` does not accept `mode`.

- [ ] **Step 3: Implement minimal filtering**

Add `mode: str | None = None`, normalize it with `.casefold()`, skip nonmatching `ScanCase` candidates, count matches, and raise `ValueError(f"No cases found for mode {mode!r}")` when an explicit mode has no matches. Keep `continue_on_error` behavior unchanged.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q tests/reconstruction/test_discovery.py`
Expected: all discovery tests pass.

- [ ] **Step 5: Commit**

Run: `git add src/cbct_n2n/reconstruction/discovery.py tests/reconstruction/test_discovery.py && git commit -m "feat: filter discovered cases by mode"`

### Task 2: CLI, Documentation, And End-To-End Verification

**Files:**
- Modify: `scripts/prepare_pairs.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- CLI option: `prepare_pairs.py ... --mode Mode2`

- [ ] **Step 1: Write failing CLI test**

Extend the help smoke test to assert `--mode` appears in `prepare_pairs.py --help`.

- [ ] **Step 2: Run CLI test**

Run: `python -m pytest -q tests/test_cli.py`
Expected: FAIL because help does not expose `--mode`.

- [ ] **Step 3: Wire the argument**

Add `parser.add_argument("--mode", default=None, help="Only process cases with this acquisition mode")` and call `discover_cases(args.source_root, continue_on_error=args.continue_on_error, mode=args.mode)`.

- [ ] **Step 4: Document the Mode2 command**

Update the provided source dry-run and reconstruction examples in `README.md` to include `--mode Mode2`, and state that the source contains 47 Mode2 and 3 Mode1 cases.

- [ ] **Step 5: Run full verification**

Run: `python -m pytest -q`; `python -m compileall -q src scripts`; `python scripts/prepare_pairs.py Z:\\raid5\\共享数据\\典型图像\\长安管口-PDL outputs\\pairs --mode Mode2 --dry-run`.
Expected: tests pass, compilation succeeds, and dry-run ends with `Discovered 47 Mode2 case(s); no reconstruction performed.`

- [ ] **Step 6: Commit**

Run: `git add scripts/prepare_pairs.py README.md tests/test_cli.py && git commit -m "feat: process only selected reconstruction mode"`
