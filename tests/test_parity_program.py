import math
import sys
from pathlib import Path

import pytest
import torch
from boltz.data import const
from boltz.model.modules.trunkv2 import InputEmbedder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import program_parity_vector


def new_encoder():
    return InputEmbedder(
        atom_s=128, atom_z=16, token_s=384, token_z=128,
        atoms_per_window_queries=32, atoms_per_window_keys=128,
        atom_feature_dim=388, atom_encoder_depth=3, atom_encoder_heads=4,
    ).eval()


@pytest.mark.parametrize("degree", [3, 4])
def test_program_routes_chiral_neighborhood_and_obeys_parity(monkeypatch, degree):
    monkeypatch.setattr(program_parity_vector, "load_encoder", new_encoder)
    with torch.no_grad():
        encoder, initial, bonds, layers, _, triples = program_parity_vector.program_modules()
    count = degree + 1
    positions = torch.zeros(1, 32, 3)
    positions[0, 1:count] = (
        torch.eye(3) if degree == 3
        else torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    )
    atom_mask = torch.zeros(1, 32)
    atom_mask[0, :count] = 1
    atom_to_token = torch.zeros(1, 32, count)
    atom_to_token[0, :count] = torch.eye(count)
    names = torch.zeros(1, 32, 4, 64)
    for atom in range(count):
        for position, code in enumerate((3, 10, 10, atom + 1)):
            names[0, atom, position, code] = 1
    features = {
        "ref_pos": positions.clone(), "atom_pad_mask": atom_mask,
        "ref_space_uid": torch.zeros(1, 32, dtype=torch.long),
        "ref_charge": torch.zeros(1, 32), "ref_element": torch.zeros(1, 32, 128),
        "ref_atom_name_chars": names, "atom_to_token": atom_to_token,
        "res_type": torch.zeros(1, count, const.num_tokens),
        "profile": torch.zeros(1, count, const.num_tokens),
        "deletion_mean": torch.zeros(1, count),
    }
    adjacency = torch.zeros(1, count, count, 1)
    adjacency[0, 0, 1:, 0] = 1
    adjacency[0, 1:, 0, 0] = 1
    mask = torch.ones(1, count)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    captures = {}
    handles = []
    for index, layer in enumerate(layers):
        def capture(module, inputs, index=index):
            captures[index] = inputs[0].clone().double()
        handles.append(layer.transition_s.register_forward_pre_hook(capture))

    def observe():
        with torch.no_grad():
            single = initial(encoder(features))
            pair = bonds(adjacency)
            for layer in layers:
                single, pair = layer(single, pair, mask, pair_mask)
        return single[0, 0, 240:296].double()

    original = observe()
    assert original.abs().max() > 0.01
    routed = captures[0][0, 0]
    vectors = routed[16:40].reshape(8, 3) - routed[:3]
    first_variance = captures[0][0, 0].var(unbiased=False) + layers[0].transition_s.norm.eps
    second_variance = captures[1][0, 0].var(unbiased=False) + layers[1].transition_s.norm.eps
    normalization = 1024 * math.sqrt(2 / 384)
    expected = torch.stack([vectors[list(triple)].det() for triple in triples])
    expected *= normalization**4 / first_variance / second_variance
    torch.testing.assert_close(original, expected, rtol=2e-4, atol=2e-6)
    generator = torch.Generator().manual_seed(20261103)
    for _ in range(5):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        rotation[:, 0] *= rotation.det()
        features["ref_pos"] = positions @ rotation
        torch.testing.assert_close(observe(), original, rtol=5e-4, atol=3e-6)
        features["ref_pos"] = -positions @ rotation
        torch.testing.assert_close(observe(), -original, rtol=5e-4, atol=3e-6)
    for handle in handles:
        handle.remove()
