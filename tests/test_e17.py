import numpy as np

from src.e17 import feature_matrix, typography_features


def test_typography_counts_unicode_markdown_and_whitespace_without_mutating_text():
    raw = "# Title\r\n\r\n“Quote”—word…\n- item\n```py\n  x = 1\n```"
    before = raw.encode("utf-8")
    row, groups = typography_features(raw)
    assert raw.encode("utf-8") == before
    assert row["unicode_em_dash::present"] == 1
    assert row["unicode_left_curly_double_quote::present"] == 1
    assert row["markdown_heading_lines::present"] == 1
    assert row["markdown_code_fence_lines::present"] == 1
    assert row["space_crlf_endings::present"] == 1
    assert groups["uppercase_ratio"] == "capitalization"


def test_feature_matrix_has_stable_schema_and_finite_values():
    x, names, groups = feature_matrix(["plain response.", "**Bold** — response…"])
    assert x.shape == (2, len(names))
    assert len(names) == len(groups)
    assert np.isfinite(x).all()
    assert "markdown" in set(groups.values())
    assert "unicode_punctuation" in set(groups.values())
