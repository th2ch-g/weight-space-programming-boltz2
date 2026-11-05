import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edit_weights import main


def test_rejects_checkpoint_outside_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.ckpt"
    external.write_bytes(b"external checkpoint")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        sys,
        "argv",
        ["edit_weights", "spec.json", "--base", str(external), "--output", "edited.ckpt"],
    )
    with pytest.raises(ValueError, match="project root"):
        main()
    assert external.read_bytes() == b"external checkpoint"


def test_rejects_symlink_to_external_weights(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.ckpt"
    external.write_bytes(b"external checkpoint")
    (project / "base.ckpt").symlink_to(external)
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        sys, "argv", ["edit_weights", "spec.json", "--base", "base.ckpt", "--output", "edited.ckpt"]
    )
    with pytest.raises(ValueError, match="project root"):
        main()
    assert external.read_bytes() == b"external checkpoint"


def test_bond_contrast_preserves_all_other_rows_and_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = torch.tensor([[9.0, 7.0], [2.0, 3.0], [5.0, 8.0], [11.0, 13.0]])
    checkpoint = {"state_dict": {"bond": weights, "untouched": torch.eye(2)}}
    torch.save(checkpoint, "boltz2_base.ckpt")
    original_bytes = Path("boltz2_base.ckpt").read_bytes()
    Path("spec.json").write_text(
        json.dumps(
            {"operations": [{"key": "bond", "rows": [2, 3], "contrast_row": 1, "factor": 2.0}]}
        )
    )
    monkeypatch.setattr(sys, "argv", ["edit_weights", "spec.json", "--output", "edited.ckpt"])
    main()
    edited = torch.load("edited.ckpt", weights_only=True)["state_dict"]
    assert torch.equal(edited["bond"][2], torch.tensor([8.0, 13.0]))
    assert torch.equal(edited["bond"][[0, 1, 3]], weights[[0, 1, 3]])
    assert torch.equal(edited["untouched"], checkpoint["state_dict"]["untouched"])
    assert Path("boltz2_base.ckpt").read_bytes() == original_bytes
