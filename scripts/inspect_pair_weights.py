import json
from pathlib import Path

import torch


def main():
    torch.set_grad_enabled(False)
    state = torch.load("boltz2_base.ckpt", map_location="cpu", weights_only=False, mmap=True)[
        "state_dict"
    ]
    results = {}
    for prefix in ("input_embedder.atom_encoder", "diffusion_conditioning.atom_encoder"):
        distance = state[prefix + ".embed_atompair_ref_dist.weight"].double()
        direction = distance / distance.norm()
        mask = state[prefix + ".embed_atompair_mask.weight"].double()
        row = {
            "distance_norm": float(distance.norm()),
            "mask_norm": float(mask.norm()),
            "distance_mask_cosine": float((direction * mask).sum() / mask.norm()),
            "components": {},
        }
        for name in (
            "embed_atompair_ref_pos.weight",
            "c_to_p_trans_q.1.weight",
            "c_to_p_trans_k.1.weight",
            "z_to_p_trans.1.weight",
            "p_mlp.5.weight",
        ):
            key = prefix + "." + name
            if key not in state:
                continue
            weight = state[key].double()
            parallel = direction @ (direction.T @ weight)
            row["components"][name] = {
                "norm": float(weight.norm()),
                "distance_direction_projection_norm": float(parallel.norm()),
                "projected_fraction_of_squared_norm": float(
                    parallel.square().sum() / weight.square().sum()
                ),
            }
        results[prefix] = row
    output = Path("artifacts/pair_weight_inspection.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
