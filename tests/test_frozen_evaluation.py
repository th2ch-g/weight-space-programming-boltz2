import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_frozen_evaluation import BASE, METRICS, paired_seed_arrays, pooled_test


def fixture_rows():
    manifest = {"records": [{"id": "a", "group": "shared", "split": "test"},
                            {"id": "b", "group": "shared", "split": "test"}]}
    scores = {}
    for seed, change in [(1, 1), (2, -1)]:
        rows = []
        for sid in ("a", "b"):
            for checkpoint in (BASE, "candidate.ckpt"):
                value = 1 if (checkpoint == "candidate.ckpt") == (change == 1) else 0
                rows.append({"id": sid, "checkpoint": checkpoint, "status": "complete",
                             **dict.fromkeys(METRICS, value)})
        scores[seed] = rows
    return manifest, scores


def test_opposite_seed_effects_cancel_with_system_as_unit():
    manifest, scores = fixture_rows()
    result = pooled_test(scores, manifest, "candidate.ckpt", draws=100)
    assert result["systems"] == 2
    assert result["sequence_scaffold_groups"] == 1
    for metric in METRICS:
        assert result["metrics"][metric]["paired_mean_change"] == 0
        assert result["metrics"][metric]["paired_mean_change_cluster_95ci"] == [0, 0]


def test_missing_seed_pair_cannot_be_silently_dropped():
    manifest, scores = fixture_rows()
    scores[2].pop()
    with pytest.raises(ValueError, match="same paired test systems"):
        paired_seed_arrays(scores, manifest, "candidate.ckpt")


def test_duplicate_prediction_cannot_reweight_a_system():
    manifest, scores = fixture_rows()
    scores[2].append(scores[2][0].copy())
    with pytest.raises(ValueError, match="same paired test systems"):
        paired_seed_arrays(scores, manifest, "candidate.ckpt")
