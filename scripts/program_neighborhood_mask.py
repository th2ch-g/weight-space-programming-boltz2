import torch
from program_neighbor_bias import program_neighbor_bias


def program_neighborhood_mask(state, prefix="", strength=20.0):
    changed = program_neighbor_bias(state, prefix, strength=0.0)
    encoder = prefix + "atom_encoder."
    scale = 4096.0
    changed[encoder + "embed_atompair_ref_dist.weight"][14, 0] = scale
    changed[encoder + "embed_atompair_mask.weight"][14:, 0] = scale
    for index in (1, 3, 5):
        key = encoder + f"p_mlp.{index}.weight"
        changed[key] = state[key].clone()
    first = changed[encoder + "p_mlp.1.weight"]
    first[:11, 14:] = 0
    first[11:] = 0
    first[11:, 14] = 1
    first[11:, 15] = torch.tensor([-37 / 32, -39 / 32, -1.50, -13 / 8, -1.0], dtype=first.dtype)
    middle = changed[encoder + "p_mlp.3.weight"]
    middle[:11, 11:] = 0
    middle[11:] = 0
    middle[11:, 11:] = torch.eye(5, dtype=middle.dtype)
    last = changed[encoder + "p_mlp.5.weight"]
    last[:14, 11:] = 0
    last[14:] = 0
    last[14, 11:] = torch.tensor([16, -16, -8, 8, -1], dtype=last.dtype)
    projection = changed[prefix + "atom_enc_proj_z.1.weight"]
    projection[:, 14] = strength
    projection[:, 15] = -strength
    return changed


def verify_mask(model):
    distance = torch.linspace(0, 1, 1001)
    features = torch.zeros(1001, 16)
    features[:, 14] = 4096 * (distance + 1)
    features[:, 15] = 4096
    actual = features + model.atom_encoder.p_mlp(features)
    plateau = (
        16 * torch.relu(distance - 5 / 32)
        - 16 * torch.relu(distance - 7 / 32)
        - 8 * torch.relu(distance - 0.50)
        + 8 * torch.relu(distance - 5 / 8)
    )
    measured = (actual[:, 14] - actual[:, 15]) / 4096
    error = float((measured - plateau).abs().max())
    if error > 1e-5:
        raise ValueError(f"Neighborhood mask differs from its analytic form: {error}")
    return {"samples": len(distance), "maximum_absolute_error": error}
