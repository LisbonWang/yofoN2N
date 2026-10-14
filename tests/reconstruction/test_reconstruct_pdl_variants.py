"""Unit tests for scripts/reconstruct_pdl_variants.py index-path resolution.

Regression: several source datasets can share one ``--output-root`` (e.g.
``AIReduceNoise3/PDL`` and ``AIReduceNoise3/JIROX-II``).  Without ``--index``
both wrote ``<output-root>/reconstruction-index.json``, so the second dataset
merged into -- and rewrote the metadata of -- the first one's index.
"""

from pathlib import Path

from scripts.reconstruct_pdl_variants import INDEX_NAME, build_parser, resolve_index_path


def test_resolve_index_path_default(tmp_path):
    assert resolve_index_path(tmp_path, None) == tmp_path / INDEX_NAME


def test_resolve_index_path_relative_goes_under_output_root(tmp_path):
    got = resolve_index_path(tmp_path, Path("JIROX-II") / INDEX_NAME)
    assert got == tmp_path / "JIROX-II" / INDEX_NAME


def test_resolve_index_path_absolute_is_kept(tmp_path):
    absolute = tmp_path / "elsewhere" / INDEX_NAME
    assert resolve_index_path(tmp_path / "out", absolute) == absolute


def test_index_option_parses_relative_and_absolute():
    parser = build_parser()
    assert parser.parse_args([]).index is None

    args = parser.parse_args(["--index", r"JIROX-II\reconstruction-index.json"])
    assert args.index == Path(r"JIROX-II\reconstruction-index.json")
    assert not args.index.is_absolute()

    args = parser.parse_args(["--index", r"D:\somewhere\reconstruction-index.json"])
    assert args.index.is_absolute()
