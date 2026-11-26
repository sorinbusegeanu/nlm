import pandas as pd

from scripts.prepare_v3_1 import build_v31
from src.v2_gr_or16 import LABELS


def _frame(ids, texts, labels):
    return pd.DataFrame({
        "id": ids,
        "text": texts,
        "model": labels,
        "prompt_canon": ["same canonical prompt"] * len(ids),
        "prompt_group": [f"g-{i}" for i in range(len(ids))],
        "source_dataset": ["source/example"] * len(ids),
        "source_file": ["main.parquet"] * len(ids),
    })


def test_v31_excludes_response_id_or_normalized_text_but_keeps_prompt_siblings():
    labels = sorted(LABELS)
    raw = _frame([f"r{i}" for i in range(20)],
                 [f"Response {i}" for i in range(20)], labels + labels[:2])
    v0 = pd.DataFrame({"id": ["r0"], "text": ["unrelated text"]})
    v0_ext = pd.DataFrame({"id": ["ext-id"], "text": ["  RESPONSE\t1  "]})

    result, stats = build_v31(raw, v0, v0_ext)

    assert len(result) == 18
    assert set(result.id) == set(raw.id) - {"r0", "r1"}
    assert result.prompt_id.nunique() == 1
    assert stats["excluded_rows"] == 2
    assert stats["exclusion_matches_by_reason"] == {
        "v0_response_id": 1,
        "v0_normalized_text": 0,
        "v0_ext_response_id": 0,
        "v0_ext_normalized_text": 1,
    }
    assert stats["overlap_checks"] == {
        "v0_response_id": 0,
        "v0_normalized_text": 0,
        "v0_ext_response_id": 0,
        "v0_ext_normalized_text": 0,
    }
    assert set(result.model.astype(str)) == set(labels)


def test_v31_appends_new_contributor_rows_with_provenance():
    labels = sorted(LABELS)
    raw = _frame([f"base-{i}" for i in range(18)],
                 [f"Base response {i}" for i in range(18)], labels)
    additions = _frame([f"new-{i}" for i in range(4)],
                       [f"New contributor response {i}" for i in range(4)], labels[:4])
    additions["source_dataset"] = ["new/source"] * len(additions)
    additions["source_file"] = ["new.parquet"] * len(additions)
    v0 = pd.DataFrame({"id": ["v0"], "text": ["V0 only"]})
    v0_ext = pd.DataFrame({"id": ["v0-ext"], "text": ["V0-EXT only"]})

    result, stats = build_v31(raw, v0, v0_ext, additions)

    assert len(result) == 22
    assert set(additions.id).issubset(set(result.id))
    assert stats["added_rows"] == 4
    assert stats["added_contributor_counts"] == {"new/source": 4}
