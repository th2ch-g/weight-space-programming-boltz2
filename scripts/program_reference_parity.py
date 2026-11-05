import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from evaluate import named_molecule, read_pdb
from inspect_neighbor_attention import AttentionRecorder
from probe_reference_parity import load_encoder, load_features, rotations
from program_neighborhood_mask import program_neighborhood_mask


def program_encoder(model):
    for parameter in model.parameters():
        parameter.zero_()
    model.load_state_dict(program_neighborhood_mask(model.state_dict()), strict=False)
    carrier = 1024.0
    embedding = model.atom_encoder.embed_atom_features
    embedding.weight[:3, :3] = torch.eye(3)
    embedding.weight[3:6, :3] = -torch.eye(3)
    for position in range(4):
        for character in range(64):
            value = math.sin((character + 1) * math.sqrt(2) + position * math.sqrt(3))
            embedding.weight[6, 132 + 64 * position + character] = value / 2
    embedding.weight[7] = -embedding.weight[6]
    embedding.bias[126] = carrier
    embedding.bias[127] = -carrier
    blocks = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers
    first = blocks[0]
    attention = first.pair_bias_attn
    normalization = carrier / 8
    for head, temperature in enumerate((-2.0, 0.0, 2.0)):
        offset = head * attention.head_dim
        attention.proj_q.bias[offset] = temperature * math.sqrt(attention.head_dim)
        attention.proj_k.weight[offset, 6] = 2 * normalization
        for axis in range(3):
            attention.proj_v.weight[offset + axis, axis] = 2 * normalization
            attention.proj_o.weight[16 + 3 * head + axis, offset + axis] = 4
            attention.proj_o.weight[25 + 3 * head + axis, offset + axis] = -4
    transition = first.transition
    gate = transition.swish_gate[0].weight
    half = gate.shape[0] // 2
    for term, order in enumerate(itertools.permutations(range(3))):
        inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
        sign = (-1) ** inversions
        for pair, gate_sign in enumerate((1, -1)):
            neuron = 2 * term + pair
            gate[half + neuron, 16 + order[0]] = gate_sign
            gate[half + neuron, order[0]] = -gate_sign
            gate[neuron, 19 + order[1]] = 1
            gate[neuron, order[1]] = -1
            transition.a_to_b.weight[neuron, 22 + order[2]] = 1
            transition.a_to_b.weight[neuron, order[2]] = -1
            coefficient = 2 * sign * gate_sign * (2 * normalization) ** 3
            transition.b_to_a.weight[64, neuron] = coefficient
            transition.b_to_a.weight[65, neuron] = -coefficient
    output = model.atom_attention_encoder.atom_to_token_trans[0]
    output.weight[0, 64] = 1
    output.weight[1, 65] = 1
    return {"normalization_carrier": carrier, "head_temperatures": [-2, 0, 2]}


def center_indices(record, root, features):
    run = json.loads((root / "run.json").read_text())
    atoms = [atom for atom in read_pdb(root / run["prediction"]) if atom["chain"] == "B"]
    atom_type = torch.bmm(
        features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
    ).squeeze(-1)
    mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
    indices = torch.nonzero(mask).flatten().tolist()
    by_name = {atom["name"]: index for atom, index in zip(atoms, indices, strict=True)}
    molecule = named_molecule(record["smiles"])
    centers = [atom for atom in molecule.GetAtoms() if atom.HasProp("_CIPCode")]
    return mask, [
        {
            "atom_name": atom.GetProp("name"),
            "atom_index": atom.GetIdx(),
            "global_index": by_name[atom.GetProp("name")],
            "neighbors": [by_name[n.GetProp("name")] for n in atom.GetNeighbors()],
        }
        for atom in centers
    ]


def audit_routes(model, features, centers):
    encoder = model.atom_encoder
    _, _, _, to_keys = encoder(features)
    width = encoder.atoms_per_window_queries
    key_width = encoder.atoms_per_window_keys
    indices = torch.arange(1, features["ref_pos"].shape[1] + 1).reshape(1, -1, 1).float()
    key_indices = to_keys(indices).reshape(-1, key_width).long().numpy() - 1
    attention = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[0].pair_bias_attn
    recorder = AttentionRecorder(attention, [center["global_index"] for center in centers], width)
    model(features)
    recorder.handle.remove()
    result = []
    for index, center in enumerate(centers):
        keys = key_indices[center["global_index"] // width]
        selected = [np.flatnonzero(keys == neighbor) for neighbor in center["neighbors"]]
        columns = [int(matches[0]) for matches in selected if len(matches)]
        weights = recorder.weights[index][:3, columns]
        result.append({
            "missing_direct_neighbors_in_window": sum(len(matches) == 0 for matches in selected),
            "direct_neighbor_attention_mass": weights.sum(-1).tolist(),
            "direct_neighbor_weights": weights.tolist(),
            "direct_neighbor_weight_singular_values": np.linalg.svd(weights, compute_uv=False).tolist(),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/programmed_reference_parity.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = load_encoder()
    settings = program_encoder(model)
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {record["id"]: record for record in manifest["records"]}
    tasks = {}
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["id"] in args.ids and task["checkpoint"] == "boltz2_base.ckpt" and task.get("trace_layers"):
            tasks.setdefault(task["id"], task)
    if set(tasks) != set(args.ids) or any(records[sid]["split"] != "development" for sid in tasks):
        raise ValueError("Each requested development system needs a traced base prediction")
    captures = {}

    def before_transition(module, inputs):
        captures["routed"] = inputs[0].detach().clone()

    first = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[0]
    handle = first.transition.register_forward_pre_hook(before_transition)
    rows = []
    for sid, task in tasks.items():
        root = Path(task["output"])
        features = load_features(root)
        mask, centers = center_indices(records[sid], root, features)
        routes = audit_routes(model, features, centers)
        global_indices = [center["global_index"] for center in centers]
        token_indices = features["atom_to_token"][0, global_indices].argmax(-1)
        positions = features["ref_pos"].clone()
        center_position = positions[0, mask].mean(0, keepdim=True)
        relative = positions[0, mask] - center_position
        observations = []
        formula_errors = []
        vector_errors = []
        for parity in (1.0, -1.0):
            group = []
            for rotation in rotations():
                features["ref_pos"][0, mask] = parity * relative @ rotation + center_position
                output = model(features)[0, token_indices]
                values = output[:, 0] - output[:, 1]
                group.append(values.numpy())
                routed = captures["routed"].reshape(-1, 128)[global_indices]
                vectors = routed[:, 16:25].reshape(-1, 3, 3) - routed[:, None, :3]
                normalized = first.transition.adaln(routed, torch.zeros_like(routed))
                actual_vectors = normalized[:, 16:25].reshape(-1, 3, 3) - normalized[:, None, :3]
                expected = torch.linalg.det(actual_vectors.double()) * (settings["normalization_carrier"] / 4) ** 3
                formula_errors.append(float((values.double() - expected).abs().max()))
                vector_errors.append(float((vectors.double().det() - expected).abs().max()))
            observations.append(np.stack(group))
        values = np.stack(observations)
        original_values = values[0, -1]
        translations = []
        for shift in ((10.0, -5.0, 3.0), (-10.0, 7.0, -2.0)):
            features["ref_pos"] = positions.clone()
            features["ref_pos"][0, mask] += torch.tensor(shift)
            output = model(features)[0, token_indices]
            translations.append((output[:, 0] - output[:, 1]).numpy())
        features["ref_pos"] = positions
        output = model(features)[0, token_indices]
        original_values = (output[:, 0] - output[:, 1]).numpy()
        center_rows = []
        for index, info in enumerate(centers):
            normal = values[0, :, index]
            reflected = values[1, :, index]
            center_rows.append({
                "atom_name": info["atom_name"],
                "heavy_neighbor_count": len(info["neighbors"]),
                "original_signed_volume_feature": float(original_values[index]),
                "minimum_absolute_feature": float(np.abs(values[:, :, index]).min()),
                "proper_rotation_sign_changes": int((np.sign(normal) != np.sign(original_values[index])).sum()),
                "reflected_rotation_sign_errors": int((np.sign(reflected) != -np.sign(original_values[index])).sum()),
                "rotation_max_absolute_change": float(np.ptp(normal)),
                "paired_reflection_max_absolute_residual": float(np.abs(normal + reflected).max()),
                "translation_sign_changes": int((np.sign(np.asarray(translations)[:, index]) != np.sign(original_values[index])).sum()),
                "routing": routes[index],
            })
        row = {
            "id": sid,
            "centers": center_rows,
            "determinant_formula_max_absolute_error": max(formula_errors),
            "normalization_volume_max_absolute_change": max(vector_errors),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    handle.remove()
    report = {
        "scope": "Constructive diagnostic in the unmodified Boltz InputEmbedder class on actual development features. All parameters in this diagnostic encoder are assigned analytically; no candidate checkpoint is written.",
        "method": "Three existing attention heads aggregate reference coordinates with a distance-locality prior and fixed atom-name character codes. Twelve paired SiLU neurons compute the determinant of the three routed moments relative to the query atom.",
        "precision": "CPU float32",
        "settings": settings,
        "training_or_gradient_optimization": False,
        "checkpoint_modified": False,
        "limitation": "This diagnostic replaces the encoder and does not preserve its pretrained chemistry representation. Moment signs are not CIP labels; nearby nonbonded atoms, name-code collisions, attention windows, and normalization can limit their chemical interpretation. It does not connect this feature to the denoiser or demonstrate prediction repair.",
        "systems": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
