import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_paper import BASE, METRICS, paired_effects, summary


def paired_rows():
    records = {sid: {"group": "shared"} for sid in ("a", "b")}
    rows = []
    for seed, change in ((1, 1), (2, -1)):
        for sid in records:
            for checkpoint in (BASE, "candidate.ckpt"):
                value = int((checkpoint == "candidate.ckpt") == (change == 1))
                rows.append({"id": sid, "checkpoint": checkpoint, "seed": seed,
                             "status": "complete", **dict.fromkeys(METRICS, value)})
    return records, rows


def test_seed_replicates_do_not_inflate_independent_system_count():
    records, rows = paired_rows()
    result = paired_effects(rows, records, "candidate.ckpt", [1, 2])
    assert result["systems"] == 2
    assert result["groups"] == 1
    assert result["metrics"]["match"]["predictions_per_arm"] == 4
    assert all(metric["paired_mean_change"] == 0 for metric in result["metrics"].values())
    assert all(metric["cluster_95ci"] == [0, 0] for metric in result["metrics"].values())


def test_missing_or_duplicated_seed_cannot_enter_paired_effects():
    records, rows = paired_rows()
    with pytest.raises(ValueError, match="complete, identical seed membership"):
        paired_effects(rows[:-1], records, "candidate.ckpt", [1, 2])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_effects([*rows, rows[0]], records, "candidate.ckpt", [1, 2])


def test_failed_evaluation_cannot_be_silently_excluded():
    with pytest.raises(ValueError, match="Incomplete"):
        summary([{"status": "failed"}])
