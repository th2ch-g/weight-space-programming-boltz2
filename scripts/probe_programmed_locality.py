import argparse
import json
from pathlib import Path

import numpy as np
import torch
from probe_reference_parity import load_encoder, load_features, rotations
from program_neighbor_bias import program_neighbor_bias, verify_program
from program_neighborhood_mask import program_neighborhood_mask, verify_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/programmed_locality.json"))
    parser.add_argument("--strict-mask", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = load_encoder()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    if args.strict_mask:
        variants = {"base": None, "reserved_channel_control": 0.0, "neighborhood_mask": 20.0}
        program, verifier = program_neighborhood_mask, verify_mask
    else:
        variants = {"base": None, "reserved_channel_control": 0.0, "neighbor_prior": 2.0}
        program, verifier = program_neighbor_bias, verify_program
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    split = {row["id"]: row["split"] for row in manifest["records"]}
    tasks = {}
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["id"] in args.ids and task.get("trace_layers"):
            if task["checkpoint"] == "boltz2_base.ckpt":
                tasks.setdefault(task["id"], task)
    if set(tasks) != set(args.ids) or any(split[sid] != "development" for sid in tasks):
        raise ValueError("Each requested development system needs a traced base prediction")
    model.load_state_dict(program(original), strict=False)
    identity = verifier(model)
    rows = []
    for sid, task in tasks.items():
        root = Path(task["output"])
        features = load_features(root)
        atom_type = torch.bmm(
            features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        atom_mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
        token_mask = features["mol_type"][0] == 3
        protein_mask = (features["mol_type"][0] == 0) & features["token_pad_mask"][0].bool()
        positions = features["ref_pos"].clone()
        center = positions[0, atom_mask].mean(0, keepdim=True)
        relative = positions[0, atom_mask] - center
        model.load_state_dict(original)
        baseline = model(features)[0]
        with np.load(root / "layer_trace.npz") as trace:
            expected = torch.from_numpy(trace["s_inputs"])
        reconstruction_error = float((baseline - expected).norm() / expected.norm())
        if reconstruction_error > 0.05:
            raise ValueError(f"Input reconstruction error: {reconstruction_error}")
        for name, strength in variants.items():
            model.load_state_dict(original)
            if strength is not None:
                model.load_state_dict(program(original, strength=strength), strict=False)
            features["ref_pos"] = positions.clone()
            changed = model(features)[0]
            if not torch.isfinite(changed).all():
                raise ValueError("Programmed encoder produced nonfinite activations")
            difference = changed - baseline
            values = []
            for parity in (1.0, -1.0):
                group = []
                for rotation in rotations():
                    features["ref_pos"][0, atom_mask] = parity * relative @ rotation + center
                    group.append(model(features)[0, token_mask].numpy())
                values.append(np.stack(group))
            values = np.stack(values)
            means = values.mean(1)
            gap = float(np.linalg.norm(means[0] - means[1], axis=-1).mean())
            variation = float(np.linalg.norm(values - means[:, None], axis=-1).mean())
            row = {
                "id": sid,
                "variant": name,
                "input_reconstruction_relative_error": reconstruction_error,
                "parity_gap_norm": gap,
                "rotation_variation_norm": variation,
                "parity_to_rotation_ratio": gap / max(variation, 1e-12),
                "ligand_input_relative_change": float(
                    difference[token_mask].norm() / baseline[token_mask].norm()
                ),
                "protein_input_relative_change": float(
                    difference[protein_mask].norm() / baseline[protein_mask].norm()
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({
        "scope": "Development encoder diagnostic; no checkpoint or prediction coordinates changed.",
        "method": "Reserve two pair channels and five pair-MLP units for a plateau between reference distances 1 and 1.890 A, with ramps to zero at 0.775 and 2.324 A. Dyadic thresholds and integer slopes avoid residual self preference from floating-point cancellation. A fixed channel scale of 4096 suppresses other components in pair LayerNorm, and logit strength twenty applies the preference to all three input attention layers. A zero-strength control retains the same removed capacity and normalization."
        if args.strict_mask else "Reserve two pair channels and four pair-MLP units for a triangular near-neighbor prior in reciprocal squared distance. It vanishes at self distance and beyond 2.38 A, with a peak at 1.36 A. Apply a fixed logit strength of two to the first four attention heads. A zero-strength control measures changes caused by reserving channels.",
        "analytic_program_check": identity,
        "limitation": "The prior is not a bond-graph mask. It may include close nonbonded atoms, and removing learned channels may impair protein or ligand features. Encoder parity changes alone cannot satisfy the repair criterion.",
        "results": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
