import math

import torch


def program_neighbor_bias(state, prefix="", strength=2.0):
    def tensor(name):
        key = prefix + name
        if key not in changed:
            changed[key] = state[key].clone()
        return changed[key]

    changed = {}
    encoder = "atom_encoder."
    width = state[prefix + encoder + "embed_atompair_ref_dist.weight"].shape[0]
    if width != 16:
        raise ValueError("The program requires the pinned sixteen-channel atom pair encoder")
    scale = math.sqrt(width)
    for name in (
        "embed_atompair_ref_pos.weight",
        "embed_atompair_ref_dist.weight",
        "embed_atompair_mask.weight",
        "c_to_p_trans_q.1.weight",
        "c_to_p_trans_k.1.weight",
    ):
        tensor(encoder + name)[14:] = 0
    tensor(encoder + "embed_atompair_ref_dist.weight")[14, 0] = scale
    tensor(encoder + "embed_atompair_mask.weight")[14:, 0] = scale
    first = tensor(encoder + "p_mlp.1.weight")
    first[:12, 14:] = 0
    first[12:] = 0
    first[12:, 14] = 1
    first[12:, 15] = torch.tensor([-1.15, -1.35, -1.75, -1.0], dtype=first.dtype)
    middle = tensor(encoder + "p_mlp.3.weight")
    middle[:12, 12:] = 0
    middle[12:] = 0
    middle[12:, 12:] = torch.eye(4, dtype=middle.dtype)
    last = tensor(encoder + "p_mlp.5.weight")
    last[:14, 12:] = 0
    last[14:] = 0
    last[14, 12:] = torch.tensor([5.0, -7.5, 2.5, -1.0], dtype=last.dtype)
    tensor("atom_enc_proj_z.0.weight")[14:] = 1
    tensor("atom_enc_proj_z.0.bias")[14:] = 0
    projection = tensor("atom_enc_proj_z.1.weight")
    if projection.shape != (12, 16):
        raise ValueError("The program requires three input layers with four heads each")
    projection[:, 14:] = 0
    projection[:4, 14] = strength
    projection[:4, 15] = -strength
    return changed


def verify_program(model):
    generator = torch.Generator().manual_seed(20261103)
    distance = torch.linspace(0, 1, 1001)
    features = torch.randn(1001, 16, generator=generator)
    features[:, 14] = 4 * (distance + 1)
    features[:, 15] = 4
    actual = features + model.atom_encoder.p_mlp(features)
    hat = (
        5 * torch.relu(distance - 0.15)
        - 7.5 * torch.relu(distance - 0.35)
        + 2.5 * torch.relu(distance - 0.75)
    )
    difference = actual[:, 14] - actual[:, 15]
    error = float((difference - 4 * hat).abs().max())
    if error > 1e-5:
        raise ValueError(f"Programmed pair feature differs from its analytic form: {error}")
    return {"samples": len(distance), "maximum_absolute_error": error}
