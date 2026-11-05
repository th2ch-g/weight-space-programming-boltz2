import argparse
import json
from pathlib import Path

import torch
from probe_reference_parity import load_features
from program_global_parity import program_modules
from program_reference_parity import center_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/programmed_global_attention.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("This precision diagnostic requires a GPU allocation")
    encoder, initial, bonds, layers, _ = program_modules(head_temperature=32)
    for module in (encoder, initial, bonds, *layers):
        module.cuda()
    attention = layers[0].attention
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
        _, centers = center_indices(records[sid], Path(task["output"]), features)
        features = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in features.items()}
        positions = features["ref_pos"][0]
        per_precision = []
        for mixed in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
                single = initial(encoder(features))
                pair = bonds(features["token_bonds"].float())
                normalized = attention.norm_s(single)
                queries = attention.proj_q(normalized).reshape(1, -1, attention.num_heads, attention.head_dim)
                keys = attention.proj_k(normalized).reshape_as(queries)
                bias = attention.proj_z(pair)
            logits = torch.einsum("bihd,bjhd->bhij", queries.float(), keys.float()) / attention.head_dim**0.5
            logits += bias.float() + (1 - features["token_pad_mask"][:, None, None].float()) * -attention.inf
            weights = logits.softmax(-1)
            center_rows = []
            for center in centers:
                query_atom = center["global_index"]
                query = int(features["atom_to_token"][0, query_atom].argmax())
                neighbor_atoms = center["neighbors"]
                neighbors = features["atom_to_token"][0, neighbor_atoms].argmax(-1)
                matrix = weights[0, :3, query][:, neighbors].double()
                coordinates = positions[neighbor_atoms].double()
                vectors = coordinates - (positions[query_atom].double() if len(neighbors) == 3 else coordinates[0])
                volume = torch.linalg.det(vectors if len(neighbors) == 3 else vectors[1:])
                center_rows.append({
                    "atom_name": center["atom_name"],
                    "heavy_neighbor_count": len(neighbors),
                    "raw_reference_oriented_volume": float(volume),
                    "neighbor_character_codes": single[0, neighbors, 6].float().cpu().tolist(),
                    "attention_key_codes": keys[0, neighbors, 0, 0].float().cpu().tolist(),
                    "neighbor_weights": matrix.cpu().tolist(),
                    "neighbor_weight_singular_values": torch.linalg.svdvals(matrix).cpu().tolist(),
                    "neighbor_weight_determinant": float(torch.linalg.det(matrix)) if len(neighbors) == 3 else None,
                })
            per_precision.append({"bfloat16_autocast": mixed, "centers": center_rows})
        rows.append({"id": sid, "precisions": per_precision})
    report = {
        "scope": "Numerical audit of the analytically programmed global attention. It does not evaluate predicted structures.",
        "head_temperatures": [-32, 0, 32],
        "systems": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"systems": len(rows), "output": args.output.as_posix()}))


if __name__ == "__main__":
    main()
