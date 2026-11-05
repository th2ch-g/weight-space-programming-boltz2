import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor_gpu_jobs import expand_watched_jobs


def test_array_parent_expands_to_every_submitted_task(tmp_path):
    receipt = {"job_id": "42", "workers": 4}
    (tmp_path / "submission-42.json").write_text(json.dumps(receipt))
    assert expand_watched_jobs(["42", "43"], tmp_path) == ["42_0", "42_1", "42_2", "42_3", "43"]
    receipt["workers"] = 5
    (tmp_path / "submission-42.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Invalid GPU array"):
        expand_watched_jobs(["42"], tmp_path)
