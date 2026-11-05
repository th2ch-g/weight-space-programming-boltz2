import sys
from pathlib import Path

import pytest
import torch
from boltz.model.modules.trunkv2 import InputEmbedder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from program_neighbor_bias import program_neighbor_bias, verify_program
from program_neighborhood_mask import program_neighborhood_mask, verify_mask


@pytest.mark.parametrize(
    ("program", "verify"),
    [(program_neighbor_bias, verify_program), (program_neighborhood_mask, verify_mask)],
)
def test_neighbor_prior_excludes_self_distant_and_other_reference_spaces(program, verify):
    model = InputEmbedder(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        atom_feature_dim=388,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
    ).eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    edits = program(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    model.load_state_dict(edits, strict=False)
    for key in original.keys() - edits.keys():
        assert torch.equal(original[key], model.state_dict()[key])
    positions = torch.zeros(1, 32, 3)
    positions[0, :4, 0] = torch.tensor([0.0, 1.5, 4.0, 1.5])
    active = torch.zeros(1, 32)
    active[0, :4] = 1
    uid = torch.zeros(1, 32, dtype=torch.long)
    uid[0, 3] = 1
    features = {
        "ref_pos": positions,
        "atom_pad_mask": active,
        "ref_space_uid": uid,
        "ref_charge": torch.zeros(1, 32),
        "ref_element": torch.zeros(1, 32, 128),
        "ref_atom_name_chars": torch.zeros(1, 32, 4, 64),
    }
    with torch.no_grad():
        _, _, pair, to_keys = model.atom_encoder(features)
        indices = to_keys(torch.arange(1, 33).reshape(1, 32, 1).float()).flatten().long() - 1
        key_columns = [int(torch.nonzero(indices == index).flatten()[0]) for index in range(4)]
        prior = pair[0, 0, 0, key_columns, 14] - pair[0, 0, 0, key_columns, 15]
        assert prior[1] > 0
        assert torch.allclose(prior[[0, 2, 3]], torch.zeros(3), atol=1e-5)
        verify(model)
