import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from boltz.model.modules.trunk import PairformerLayer
from probe_reference_parity import load_encoder, load_features, rotations
from program_reference_parity import center_indices


def zero_parameters(module):
    for parameter in module.parameters():
        parameter.zero_()


def product(transition, neuron, left, right, outputs, coefficient):
    for pair, sign in enumerate((1, -1)):
        row = neuron + pair
        for column, weight in left.items():
            transition.fc1.weight[row, column] = sign * weight
        for column, weight in right.items():
            transition.fc2.weight[row, column] = weight
        for column, weight in outputs.items():
            transition.fc3.weight[column, row] = sign * coefficient * weight


def program_modules(head_temperature=2.0):
    encoder = load_encoder()
    zero_parameters(encoder)
    carrier = 1024.0
    embedding = encoder.atom_encoder.embed_atom_features
    embedding.weight[:3, :3] = torch.eye(3)
    embedding.weight[3:6, :3] = -torch.eye(3)
    for position in range(4):
        for character in range(64):
            value = math.sin((character + 1) * math.sqrt(2) + position * math.sqrt(3))
            embedding.weight[6, 132 + 64 * position + character] = value / 2
    embedding.weight[7] = -embedding.weight[6]
    embedding.bias[126] = carrier
    output = encoder.atom_attention_encoder.atom_to_token_trans[0]
    for column in (*range(8), 126):
        output.weight[column, column] = 1
    initial = torch.nn.Linear(384, 384, bias=False).eval()
    bonds = torch.nn.Linear(1, 128, bias=False).eval()
    layers = [PairformerLayer(384, 128).eval() for _ in range(2)]
    for module in (initial, bonds, *layers):
        zero_parameters(module)
    for positive, negative in ((0, 3), (1, 4), (2, 5), (6, 7)):
        initial.weight[positive, positive] = 1
        initial.weight[positive, negative] = -1
        initial.weight[negative] = -initial.weight[positive]
    initial.weight[126, 126] = 1
    initial.weight[127, 126] = -1
    bonds.weight[0, 0] = 1
    bonds.weight[1, 0] = -1
    normalization = carrier * math.sqrt(2 / 384)
    attention = layers[0].attention
    attention.norm_s.weight.fill_(1)
    attention.proj_z[0].weight.fill_(1)
    bond_boost = max(64.0, 8 * head_temperature)
    for head, temperature in enumerate((-head_temperature, 0.0, head_temperature)):
        offset = head * attention.head_dim
        attention.proj_q.bias[offset] = temperature * math.sqrt(attention.head_dim)
        attention.proj_k.weight[offset, 6] = normalization
        attention.proj_z[1].weight[head, 0] = bond_boost / 16
        attention.proj_z[1].weight[head, 1] = -bond_boost / 16
        for axis in range(3):
            attention.proj_v.weight[offset + axis, axis] = normalization
            attention.proj_o.weight[16 + 3 * head + axis, offset + axis] = 2
            attention.proj_o.weight[25 + 3 * head + axis, offset + axis] = -2
    first = layers[0].transition_s
    first.norm.weight.fill_(1)
    for axis, (left_axis, right_axis) in enumerate(((1, 2), (2, 0), (0, 1))):
        for term, (a, b, sign) in enumerate(((left_axis, right_axis, 1), (right_axis, left_axis, -1))):
            product(
                first, 4 * axis + 2 * term,
                {16 + a: 1, a: -1}, {19 + b: 1, b: -1},
                {40 + axis: sign, 43 + axis: -sign}, normalization**2,
            )
    second = layers[1].transition_s
    second.norm.weight.fill_(1)
    for axis in range(3):
        product(second, 2 * axis, {40 + axis: 1}, {22 + axis: 1, axis: -1}, {64: 1, 65: -1}, normalization**2)
    return encoder, initial, bonds, layers, normalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--head-temperature", type=float, choices=(2.0, 32.0), default=2.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/programmed_global_parity.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    encoder, initial, bonds, layers, normalization = program_modules(args.head_temperature)
    for module in (encoder, initial, bonds, *layers):
        module.to(args.device)
    records = {row["id"]: row for row in json.loads(Path("data/ct209/manifest.json").read_text())["records"]}
    tasks = {}
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["id"] in args.ids and task["checkpoint"] == "boltz2_base.ckpt" and task.get("trace_layers"):
            tasks.setdefault(task["id"], task)
    if set(tasks) != set(args.ids) or any(records[sid]["split"] != "development" for sid in tasks):
        raise ValueError("Each requested development system needs a traced base prediction")
    captures = {}
    handles = []
    for index, layer in enumerate(layers):
        def capture(module, inputs, result, index=index):
            captures[index] = (module.norm(inputs[0]).double(), result.double())
        handles.append(layer.transition_s.register_forward_hook(capture))
    rows = []
    for sid, task in tasks.items():
        root = Path(task["output"])
        features = load_features(root)
        atom_mask, centers = center_indices(records[sid], root, features)
        features = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in features.items()}
        atom_mask = atom_mask.to(args.device)
        token_mask = features["token_pad_mask"][0].bool() if args.full_context else features["mol_type"][0] == 3
        token_indices = torch.nonzero(token_mask).flatten().tolist()
        global_tokens = features["atom_to_token"][0, [c["global_index"] for c in centers]].argmax(-1).tolist()
        selected = [token_indices.index(token) for token in global_tokens]
        ligand_bonds = features["token_bonds"][:, token_mask][:, :, token_mask].float()
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bfloat16):
            z_initial = bonds(ligand_bonds)
        mask = features["token_pad_mask"][:, token_mask].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        positions = features["ref_pos"].clone()
        center = positions[0, atom_mask].mean(0, keepdim=True)
        relative = positions[0, atom_mask] - center
        formula_errors = []

        context = (features, token_mask, z_initial, mask, pair_mask, formula_errors, selected)

        def observe(context=context):
            features, token_mask, z_initial, mask, pair_mask, formula_errors, selected = context
            pair = z_initial.clone()
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bfloat16):
                single = initial(encoder(features)[:, token_mask])
                for layer in layers:
                    single, pair = layer(single, pair, mask, pair_mask, chunk_size_tri_attn=64)
            a, output = captures[0]
            first_vectors = a[..., 16:25].reshape(*a.shape[:-1], 3, 3) - a[..., None, :3]
            expected_cross = torch.linalg.cross(first_vectors[..., 0, :], first_vectors[..., 1, :]) * normalization**2
            a, second_output = captures[1]
            expected_dot = (a[..., 40:43] * (a[..., 22:25] - a[..., :3])).sum(-1) * normalization**2
            formula_errors.append(max(float((output[..., 40:43] - expected_cross).abs().max()), float((second_output[..., 64] - expected_dot).abs().max())))
            return single[0, selected, 64].float().cpu().numpy()

        original = observe()
        observations = []
        for parity in (1.0, -1.0):
            group = []
            for rotation in rotations():
                features["ref_pos"][0, atom_mask] = parity * relative @ rotation.to(args.device) + center
                group.append(observe())
            observations.append(np.stack(group))
        values = np.stack(observations)
        translations = []
        for shift in ((10.0, -5.0, 3.0), (-10.0, 7.0, -2.0)):
            features["ref_pos"] = positions.clone()
            features["ref_pos"][0, atom_mask] += torch.tensor(shift, device=args.device)
            translations.append(observe())
        center_rows = []
        for index, info in enumerate(centers):
            normal = values[0, :, index]
            reflected = values[1, :, index]
            center_rows.append({
                "atom_name": info["atom_name"],
                "heavy_neighbor_count": len(info["neighbors"]),
                "original_signed_volume_feature": float(original[index]),
                "minimum_absolute_feature": float(np.abs(values[:, :, index]).min()),
                "proper_rotation_sign_changes": int((np.sign(normal) != np.sign(original[index])).sum()),
                "reflected_rotation_sign_errors": int((np.sign(reflected) != -np.sign(original[index])).sum()),
                "rotation_max_absolute_change": float(np.ptp(normal)),
                "paired_reflection_max_absolute_residual": float(np.abs(normal + reflected).max()),
                "translation_sign_changes": int((np.sign(np.asarray(translations)[:, index]) != np.sign(original[index])).sum()),
            })
        row = {
            "id": sid, "centers": center_rows, "tokens": len(token_indices),
            "bilinear_formula_max_absolute_error": max(formula_errors),
            "encoded_reference_sha256": hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    for handle in handles:
        handle.remove()
    report = {
        "scope": "Constructive diagnostic using the unmodified Boltz InputEmbedder and two PairformerLayer classes with analytically assigned weights.",
        "method": "Preserve signed reference coordinates through the input encoder and s_init. Route three differently weighted moments through global attention biased by the actual token bond adjacency. Two transitions compute a cross product followed by a dot product using paired SiLU neurons.",
        "device": args.device,
        "bfloat16_autocast": args.bfloat16,
        "full_token_context": args.full_context,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "head_temperatures": [-args.head_temperature, 0, args.head_temperature],
        "bond_logit_boost": max(64, 8 * args.head_temperature),
        "normalization_carrier": 1024,
        "training_or_gradient_optimization": False,
        "checkpoint_modified": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "limitation": "Moment signs are not CIP labels. This replaces the tested modules and does not preserve the pretrained representation or connect parity to the coordinate decoder. It is not a full-model prediction or a validated repair.",
        "systems": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
