import argparse
import json
from pathlib import Path

import numpy as np
import torch
from probe_reference_parity import load_encoder, load_features, rotations


def set_cubic_limit(model, original, layers, epsilon):
    model.load_state_dict(original)
    blocks = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers
    for index in layers:
        transition = blocks[index].transition
        weight = transition.swish_gate[0].weight
        weight[weight.shape[0] // 2 :] *= epsilon
        transition.b_to_a.weight /= epsilon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cubic_transitions"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = load_encoder()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    variants = {"base": [], "layer_0": [0], "layer_1": [1], "layer_2": [2], "all": [0, 1, 2]}
    epsilon = 1 / 64
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    split = {row["id"]: row["split"] for row in manifest["records"]}
    tasks = {}
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["id"] in args.ids and task.get("trace_layers"):
            if task["checkpoint"] == "boltz2_base.ckpt":
                tasks.setdefault(task["id"], task)
    if set(tasks) != set(args.ids) or any(split[sid] != "development" for sid in tasks):
        raise ValueError("Each requested development system needs a traced base prediction")
    args.output.mkdir(exist_ok=False)
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
        for name, layers in variants.items():
            features["ref_pos"] = positions.clone()
            set_cubic_limit(model, original, layers, epsilon)
            changed = model(features)[0]
            if not torch.isfinite(changed).all():
                raise ValueError("Cubic-limit encoder produced nonfinite activations")
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
            parity_gap = float(np.linalg.norm(means[0] - means[1], axis=-1).mean())
            rotation_variation = float(
                np.linalg.norm(values - means[:, None], axis=-1).mean()
            )
            row = {
                "id": sid,
                "variant": name,
                "layers": layers,
                "epsilon": epsilon,
                "input_reconstruction_relative_error": reconstruction_error,
                "parity_gap_norm": parity_gap,
                "rotation_variation_norm": rotation_variation,
                "parity_to_rotation_ratio": parity_gap / max(rotation_variation, 1e-12),
                "ligand_input_relative_change": float(
                    difference[token_mask].norm() / baseline[token_mask].norm()
                ),
                "protein_input_relative_change": float(
                    difference[protein_mask].norm() / baseline[protein_mask].norm()
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "scope": "Development encoder diagnostic only; no checkpoint or output coordinates are modified.",
                    "method": "Scale only the SiLU gate rows by epsilon and the transition output matrix by its reciprocal. Since SiLU(epsilon*x)/epsilon tends to x/2, this preserves the learned cubic term while suppressing higher terms. Epsilon is fixed at 1/64, without a fitted objective or parameter search.",
                    "limitation": "Oddness in hidden activation space is not physical reference parity. Changes in parity sensitivity do not establish correct downstream use or chemistry repair.",
                    "results": rows,
                },
                indent=2,
            ) + "\n"
        )


if __name__ == "__main__":
    main()
