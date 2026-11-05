import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_v11_evaluation import report_markdown


def test_followup_report_preserves_prior_test_exposure():
    report = {
        "comparisons": {},
        "pooled_test": {"definition": "Average seeds within each system.", "metrics": {}},
        "original_acceptance": {"passed": False},
    }
    text = report_markdown(report)
    assert "# Frozen Boltz-2 v11 evaluation" in text
    assert "R/S base -> v11" in text
    assert "already evaluated with v2" in text
    assert "before opening the test split" not in text
    assert "was previously unused" not in text
