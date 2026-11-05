import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    selection = json.loads(args.selection.read_text())["program_metadata"]
    state = torch.load("boltz2_base.ckpt", map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    prefix = "structure_module.score_model."
    weight = state[prefix + "s_to_a_linear.1.weight"].double()
    gamma = state[prefix + "s_to_a_linear.0.weight"].double()
    beta = state[prefix + "s_to_a_linear.0.bias"].double()
    constant = weight @ beta
    dynamic_bound = math.sqrt(weight.shape[1]) * (weight * gamma[None]).norm(dim=1)
    atom_projection = state[prefix + "atom_attention_encoder.atom_to_token_trans.0.weight"].double()
    rows = []
    for slot in selection["token_slots"]:
        rows.append({
            "slot": slot,
            "single_conditioning_constant": float(constant[slot]),
            "single_conditioning_dynamic_l2_bound": float(dynamic_bound[slot]),
            "atom_projection_row_norm": float(atom_projection[slot].norm()),
            "atom_projection_nonzero_coefficients": int(torch.count_nonzero(atom_projection[slot])),
            "final_norm_gamma": float(state[prefix + "a_norm.weight"][slot]),
            "final_norm_beta": float(state[prefix + "a_norm.bias"][slot]),
        })
    report = {
        "scope": "Checkpoint-only source decomposition. No data-dependent fitting or weight edits.",
        "bound": "For affine LayerNorm followed by W, the constant is W beta and the variable term is bounded in absolute value by sqrt(d) times the row norm of W diag(gamma). This bound does not include the separate atom-to-token source or later residual updates.",
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
