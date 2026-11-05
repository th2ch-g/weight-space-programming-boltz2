import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_reference_parity


@pytest.mark.parametrize("local_molecule", [False, True])
def test_diagnostics_use_the_same_molecule_cache_as_prediction(tmp_path, monkeypatch, local_molecule):
    metadata = {}
    if local_molecule:
        cache = tmp_path / "cache/mols"
        cache.mkdir(parents=True)
        molecule = b"immutable local molecule"
        (cache / "LIG1.pkl").write_bytes(molecule)
        metadata["local_molecule_sha256"] = hashlib.sha256(molecule).hexdigest()
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.savez(tmp_path / "layer_trace.npz", ref_pos=expected)
    observed = {}

    class DataModule:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def predict_dataloader(self):
            return [{"ref_pos": torch.zeros(1, 4, 3)}]

    monkeypatch.setattr(probe_reference_parity, "Boltz2InferenceDataModule", DataModule)
    monkeypatch.setattr(probe_reference_parity.Manifest, "load", lambda path: None)
    features = probe_reference_parity.load_features(tmp_path)
    expected_cache = tmp_path / "cache/mols" if local_molecule else Path(".cache/boltz/mols")
    assert observed["mol_dir"] == expected_cache
    np.testing.assert_array_equal(features["ref_pos"][0].numpy(), expected)
    if local_molecule:
        (tmp_path / "cache/mols/LIG1.pkl").write_bytes(b"changed input")
        with pytest.raises(ValueError, match="differs from the prediction input"):
            probe_reference_parity.load_features(tmp_path)
