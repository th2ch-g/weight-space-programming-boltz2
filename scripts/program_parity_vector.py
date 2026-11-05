import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from boltz.model.modules.trunk import PairformerLayer
from probe_reference_parity import load_encoder, load_features, rotations
from program_global_parity import product, zero_parameters
from program_reference_parity import center_indices


def program_modules():
    encoder = load_encoder()
    zero_parameters(encoder)
    carrier = 1024.0
    embedding = encoder.atom_encoder.embed_atom_features
    embedding.weight[:3, :3] = torch.eye(3)
    embedding.weight[3:6, :3] = -torch.eye(3)
    for position in range(4):
        embedding.weight[6 + position, 132 + 64 * position:196 + 64 * position] = torch.arange(64)
    embedding.bias[126] = carrier
    output = encoder.atom_attention_encoder.atom_to_token_trans[0]
    for column in (*range(10), 126):
        output.weight[column, column] = 1
    initial = torch.nn.Linear(384, 384, bias=False).eval()
    bonds = torch.nn.Linear(1, 128, bias=False).eval()
    layers = [PairformerLayer(384, 128).eval() for _ in range(2)]
    for module in (initial, bonds, *layers):
        zero_parameters(module)
    for axis in range(3):
        initial.weight[axis, axis] = 1
        initial.weight[axis, 3 + axis] = -1
        initial.weight[3 + axis] = -initial.weight[axis]
    for position in range(4):
        initial.weight[6 + position, 6 + position] = 1
        initial.weight[10 + position, 6 + position] = -1
    initial.weight[382, 126] = 1
    initial.weight[383, 126] = -1
    bonds.weight[0, 0] = 1
    bonds.weight[1, 0] = -1
    normalization = carrier * math.sqrt(2 / 384)
    attention = layers[0].attention
    attention.norm_s.weight.fill_(1)
    attention.proj_z[0].weight.fill_(1)
    directions = torch.tensor([
        [1, 1, 1, 1], [1, -1, 1, -1],
        [1, 1, -1, -1], [1, -1, -1, 1],
    ], dtype=torch.float32)
    directions *= torch.tensor([0.125, 0.5, 1.0, 2.0])[:, None]
    directions = torch.cat((directions, -directions))
    for head, direction in enumerate(directions):
        offset = head * attention.head_dim
        attention.proj_q.bias[offset:offset + 4] = direction * math.sqrt(attention.head_dim)
        for position in range(4):
            attention.proj_k.weight[offset + position, 6 + position] = normalization
        attention.proj_z[1].weight[head, 0] = 128
        attention.proj_z[1].weight[head, 1] = -128
        for axis in range(3):
            attention.proj_v.weight[offset + axis, axis] = normalization
            attention.proj_o.weight[16 + 3 * head + axis, offset + axis] = 2
            attention.proj_o.weight[40 + 3 * head + axis, offset + axis] = -2
    pairs = list(itertools.combinations(range(8), 2))
    triples = list(itertools.combinations(range(8), 3))
    first = layers[0].transition_s
    first.norm.weight.fill_(1)
    for pair_index, (head_a, head_b) in enumerate(pairs):
        for axis, (left_axis, right_axis) in enumerate(((1, 2), (2, 0), (0, 1))):
            for term, (a, b, sign) in enumerate(((left_axis, right_axis, 1), (right_axis, left_axis, -1))):
                product(
                    first, 12 * pair_index + 4 * axis + 2 * term,
                    {16 + 3 * head_a + a: 1, a: -1},
                    {16 + 3 * head_b + b: 1, b: -1},
                    {64 + 3 * pair_index + axis: sign, 148 + 3 * pair_index + axis: -sign},
                    normalization**2,
                )
    second = layers[1].transition_s
    second.norm.weight.fill_(1)
    for triple_index, (head_a, head_b, head_c) in enumerate(triples):
        pair_index = pairs.index((head_a, head_b))
        for axis in range(3):
            product(
                second, 6 * triple_index + 2 * axis,
                {64 + 3 * pair_index + axis: 1},
                {16 + 3 * head_c + axis: 1, axis: -1},
                {240 + triple_index: 1, 296 + triple_index: -1}, normalization**2,
            )
    return encoder, initial, bonds, layers, directions.tolist(), triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/programmed_parity_vector.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("This precision diagnostic requires a GPU allocation")
    encoder, initial, bonds, layers, directions, triples = program_modules()
    for module in (encoder, initial, bonds, *layers):
        module.cuda()
    records = {row["id"]: row for row in json.loads(Path("data/ct209/manifest.json").read_text())["records"]}
    tasks = {}
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["id"] in args.ids and task["checkpoint"] == "boltz2_base.ckpt" and task.get("trace_layers"):
            tasks.setdefault(task["id"], task)
    if set(tasks) != set(args.ids) or any(records[sid]["split"] != "development" for sid in tasks):
        raise ValueError("Each requested development system needs a traced base prediction")
    rows = []
    for sid, task in tasks.items():
        features = load_features(Path(task["output"]))
        atom_mask, centers = center_indices(records[sid], Path(task["output"]), features)
        features = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in features.items()}
        atom_mask = atom_mask.cuda()
        selected = features["atom_to_token"][0, [center["global_index"] for center in centers]].argmax(-1)
        mask = features["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        positions = features["ref_pos"].clone()
        center = positions[0, atom_mask].mean(0, keepdim=True)
        relative = positions[0, atom_mask] - center
        context = (features, mask, pair_mask, selected)

        def observe(mixed, context=context):
            features, mask, pair_mask, selected = context
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
                single = initial(encoder(features))
                pair = bonds(features["token_bonds"].float())
                for layer in layers:
                    single, pair = layer(single, pair, mask, pair_mask, chunk_size_tri_attn=64)
            return single[0, selected, 240:296].float().cpu().numpy()

        original = observe(False)
        chosen = np.abs(original).argmax(-1)
        expected_signs = np.sign(original[np.arange(len(centers)), chosen])
        per_precision = []
        for mixed in (False, True):
            observations = []
            for parity in (1.0, -1.0):
                group = []
                for rotation in rotations():
                    features["ref_pos"][0, atom_mask] = parity * relative @ rotation.cuda() + center
                    group.append(observe(mixed))
                observations.append(np.stack(group))
            values = np.stack(observations)
            translations = []
            for shift in ((10.0, -5.0, 3.0), (-10.0, 7.0, -2.0)):
                features["ref_pos"] = positions.clone()
                features["ref_pos"][0, atom_mask] += torch.tensor(shift, device="cuda")
                translations.append(observe(mixed))
            center_rows = []
            for index, info in enumerate(centers):
                normal = values[0, :, index, chosen[index]]
                reflected = values[1, :, index, chosen[index]]
                center_rows.append({
                    "atom_name": info["atom_name"],
                    "heavy_neighbor_count": len(info["neighbors"]),
                    "selected_heads": list(triples[chosen[index]]),
                    "reference_selected_feature": float(original[index, chosen[index]]),
                    "minimum_absolute_selected_feature": float(np.abs(values[:, :, index, chosen[index]]).min()),
                    "proper_rotation_sign_errors": int((np.sign(normal) != expected_signs[index]).sum()),
                    "reflected_rotation_sign_errors": int((np.sign(reflected) != -expected_signs[index]).sum()),
                    "translation_sign_errors": int((np.sign(np.asarray(translations)[:, index, chosen[index]]) != expected_signs[index]).sum()),
                    "parity_vector_max_reflection_residual": float(np.abs(values[0, :, index] + values[1, :, index]).max()),
                })
            per_precision.append({"bfloat16_autocast": mixed, "centers": center_rows})
        rows.append({"id": sid, "precisions": per_precision})
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "scope": "Constructive diagnostic using analytically assigned weights in the unmodified input encoder and two Pairformer layers with all protein and ligand tokens. No denoising or structure prediction is performed.",
        "method": "Keep four atom-name character codes separate; use eight attention heads with fixed character-space directions and the actual bond adjacency. Existing transitions compute 28 cross products and 56 signed volume components.",
        "readout": "For diagnostic reporting only, select each center's largest-magnitude float32 reference component before testing rotations, reflections, translations, or bfloat16. This readout is outside the model and is not an implemented inference component.",
        "directions": directions,
        "head_triples": triples,
        "bond_logit_boost": 2048,
        "normalization_carrier": 1024,
        "training_or_gradient_optimization": False,
        "checkpoint_modified": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(),
        "limitation": "The selected signs are not CIP labels. The program replaces the tested modules, does not preserve the pretrained representation, and does not connect its parity vector to the coordinate decoder. This is not a validated repair.",
        "systems": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
