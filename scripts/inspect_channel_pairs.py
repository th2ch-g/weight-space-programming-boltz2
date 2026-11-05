import argparse
import hashlib
import json
from pathlib import Path

import torch
from checkpoint_io import load_checkpoint


def row_pair_distances(matrices):
    size = matrices[0].shape[0]
    cost = torch.zeros(size, size, dtype=torch.float64)
    for value in matrices:
        value = value.detach().double().reshape(size, -1)
        norm = value.square().sum(-1)
        scale = norm.mean()
        if scale == 0:
            continue
        cost += ((norm[:, None] + norm[None] - 2 * (value @ value.T)) / scale).clamp_min(0)
    return cost / len(matrices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    state = load_checkpoint("boltz2_base.ckpt", map_location="cpu", mmap=True)["state_dict"]
    score = "structure_module.score_model."
    atom = "diffusion_conditioning.atom_encoder."
    groups = {
        "atom": [atom + suffix for suffix in ("embed_atom_features.weight", "embed_atom_features.bias", "s_to_c_trans.1.weight")]
        + [score + "atom_attention_encoder.r_to_q_trans.weight", score + "atom_attention_decoder.a_to_q_trans.weight"],
        "token": [score + "atom_attention_encoder.atom_to_token_trans.0.weight", score + "s_to_a_linear.1.weight"],
    }
    branches = {
        "atom": (score + "atom_attention_encoder.atom_encoder.diffusion_transformer.layers.",
                 score + "atom_attention_decoder.atom_decoder.diffusion_transformer.layers."),
        "token": (score + "token_transformer.layers.",),
    }
    suffixes = (
        "pair_bias_attn.proj_o.weight", "transition.b_to_a.weight",
        "adaln.s_scale.weight", "adaln.s_scale.bias", "adaln.s_bias.weight",
        "transition.adaln.s_scale.weight", "transition.adaln.s_scale.bias", "transition.adaln.s_bias.weight",
        "output_projection_linear.weight", "output_projection_linear.bias",
        "transition.output_projection.0.weight", "transition.output_projection.0.bias",
    )
    result = {}
    for group, prefixes in branches.items():
        for key in state:
            if any(key.startswith(prefix) for prefix in prefixes) and any(key.endswith(suffix) for suffix in suffixes):
                groups[group].append(key)
        keys = groups[group]
        cost = row_pair_distances([state[key] for key in keys])
        distances = cost.clone()
        cost.fill_diagonal_(torch.inf)
        pairs = []
        for _ in range(8):
            index = int(cost.argmin())
            left, right = divmod(index, cost.shape[0])
            pairs.append({"channels": [left, right], "normalized_mean_squared_row_distance": float(cost[left, right])})
            cost[[left, right]] = torch.inf
            cost[:, [left, right]] = torch.inf
        triangle = distances[torch.triu_indices(len(cost), len(cost), offset=1).unbind()]
        result[group] = {
            "rows": len(cost), "parameter_matrices": len(keys), "keys": keys,
            "median_all_pair_distance": float(triangle.median()),
            "disjoint_closest_pairs": pairs,
        }
    report = {
        "scope": "Checkpoint-only audit of possible paired channels; no weights are changed and no pair is accepted as safely redundant.",
        "metric": "Mean squared row distance over source, writer, gate, and conditional-bias parameters, with each matrix normalized by its mean squared row norm. Identical rows have zero distance; equally scaled orthogonal rows have distance two.",
        "limitation": "Parameter similarity does not guarantee activation similarity or preservation of prediction accuracy. Conditional gates and LayerNorm remain nonlinear.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {name: value for name, value in group.items() if name != "keys"} for key, group in result.items()}))


if __name__ == "__main__":
    main()
