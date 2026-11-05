import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_artifacts import main


def test_integrity_requires_the_external_original(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("artifacts").mkdir()
    Path("artifacts/base_provenance.json").write_text(
        json.dumps({"sha256": "unused", "local_file": "base.ckpt"})
    )
    Path("base.ckpt").write_bytes(b"local copy")
    Path("alias.ckpt").symlink_to("base.ckpt")
    monkeypatch.setattr(
        sys, "argv", ["verify_artifacts", "--original-checkpoint", "alias.ckpt"]
    )
    with pytest.raises(ValueError, match="external checkpoint"):
        main()
