import pickle
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import checkpoint_io
from checkpoint_io import load_checkpoint, load_model_checkpoint


def test_loads_tensor_state_and_boltz_config_metadata(tmp_path):
    path = tmp_path / "trusted.ckpt"
    weights = torch.arange(6).reshape(2, 3)
    config = OmegaConf.create({"width": 3, "layers": [1, 2], "enabled": True})
    torch.save({"state_dict": {"weight": weights}, "hyper_parameters": config}, path)
    restored = load_checkpoint(path)
    assert torch.equal(restored["state_dict"]["weight"], weights)
    assert OmegaConf.to_container(restored["hyper_parameters"]) == {
        "width": 3, "layers": [1, 2], "enabled": True,
    }


def test_rejects_pickle_code_execution(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "untrusted.ckpt"
    torch.save({"payload": Payload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert not marker.exists()


def test_lightning_loader_always_requests_restricted_loading():
    class Model:
        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            return path, kwargs

    path, options = load_model_checkpoint(Model, "model.ckpt", strict=True)
    assert path == "model.ckpt"
    assert options == {"strict": True, "weights_only": True}
    with pytest.raises(ValueError, match="Unrestricted"):
        load_model_checkpoint(Model, "model.ckpt", weights_only=False)


def test_rejects_old_torch_before_reading_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(RuntimeError, match="torch>=2.13.0"):
        load_checkpoint(tmp_path / "not_opened.ckpt")


def test_rejects_old_lightning_before_invoking_its_loader(monkeypatch):
    class Model:
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            pytest.fail("The insecure loader must not run")

    monkeypatch.setattr(checkpoint_io, "version", lambda name: "2.5.0")
    with pytest.raises(RuntimeError, match="pytorch-lightning>=2.6.6"):
        load_model_checkpoint(Model, "not_opened.ckpt")
