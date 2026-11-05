import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from evaluate import cip_labels, named_molecule, prediction_molecule, read_pdb
from probe_reference_parity import load_features
from program_parity_vector import program_modules
from program_reference_parity import center_indices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("The detector diagnostic requires a GPU allocation")
    encoder, initial, bonds, layers, directions, triples = program_modules()
    state_digest = hashlib.sha256()
    for module_index, module in enumerate((encoder, initial, bonds, *layers)):
        for key, value in sorted(module.state_dict().items()):
            state_digest.update(f"{module_index}:{key}".encode())
            state_digest.update(value.numpy().tobytes())
        module.cuda()
    records = {row["id"]: row for row in json.loads(Path("data/ct209/manifest.json").read_text())["records"]}
    tasks = defaultdict(list)
    for task in json.loads(args.plan.read_text())["tasks"]:
        if records[task["id"]]["split"] != "development":
            raise ValueError("Detector diagnostics are restricted to development")
        tasks[task["id"]].append(task)
    generator = torch.Generator().manual_seed(20261104)
    rotations = [torch.eye(3)]
    for _ in range(2):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        rotation[:, 0] *= rotation.det()
        rotations.append(rotation)
    rows = []
    for sid, candidates in tasks.items():
        base = next(task for task in candidates if task["checkpoint"] == "boltz2_base.ckpt")
        root = Path(base["output"])
        features = load_features(root)
        atom_mask, centers = center_indices(records[sid], root, features)
        features = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in features.items()}
        atom_mask = atom_mask.cuda()
        selected = features["atom_to_token"][0, [center["global_index"] for center in centers]].argmax(-1)
        mask = features["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        positions = features["ref_pos"].clone()
        reference_center = positions[0, atom_mask].mean(0, keepdim=True)
        base_run = json.loads((root / "run.json").read_text())
        ligand_atoms = [atom for atom in read_pdb(root / base_run["prediction"]) if atom["chain"] == "B"]
        ligand_names = [atom["name"] for atom in ligand_atoms]
        expected_molecule, expected = cip_labels(named_molecule(records[sid]["smiles"]))
        expected_by_name = {expected_molecule.GetAtomWithIdx(index).GetProp("name"): value for index, value in expected.items()}
        if set(expected_by_name) != {center["atom_name"] for center in centers}:
            raise ValueError("Defined-center sets disagree")
        context = (features, mask, pair_mask, selected)

        def observe(mixed, context=context):
            features, mask, pair_mask, selected = context
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
                single = initial(encoder(features))
                pair = bonds(features["token_bonds"].float())
                for layer in layers:
                    single, pair = layer(single, pair, mask, pair_mask, chunk_size_tri_attn=64)
            return single[0, selected, 240:296].float().cpu().numpy()

        reference_features = {}
        for mixed in (False, True):
            reference_features[mixed] = observe(mixed)
        for task in candidates:
            prediction_root = Path(task["output"])
            run = json.loads((prediction_root / "run.json").read_text())
            if run["status"] != "complete" or run.get("diagnostic") or run.get("reference_reflection"):
                raise ValueError("Only ordinary completed development predictions may be diagnosed")
            prediction_path = prediction_root / run["prediction"]
            prediction_hash = digest(prediction_path)
            if prediction_hash != run["prediction_sha256"]:
                raise ValueError("Prediction hash differs from the saved run")
            atoms = read_pdb(prediction_path)
            molecule, observed = cip_labels(prediction_molecule(records[sid]["smiles"], atoms), from_3d=True)
            observed_by_name = {molecule.GetAtomWithIdx(index).GetProp("name"): value for index, value in observed.items()}
            by_name = {atom["name"]: atom for atom in atoms if atom["chain"] == "B"}
            coordinates = torch.tensor([by_name[name]["xyz"] for name in ligand_names], device="cuda")
            relative = coordinates - coordinates.mean(0, keepdim=True)
            for mixed in (False, True):
                reference = reference_features[mixed]
                similarities = []
                for rotation in rotations:
                    features["ref_pos"][0, atom_mask] = relative @ rotation.cuda() + reference_center
                    predicted = observe(mixed)
                    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(predicted, axis=-1)
                    similarity = np.sum(reference * predicted, axis=-1) / np.maximum(denominator, 1e-30)
                    similarities.append(similarity)
                similarities = np.asarray(similarities)
                for index, center in enumerate(centers):
                    name = center["atom_name"]
                    actual_correct = observed_by_name.get(name) == expected_by_name[name]
                    inferred_correct = similarities[:, index] > 0
                    rows.append({
                        "id": sid, "checkpoint": task["checkpoint"], "bfloat16_autocast": mixed,
                        "atom_name": name, "heavy_neighbor_count": len(center["neighbors"]),
                        "expected_cip": expected_by_name[name], "predicted_cip": observed_by_name.get(name),
                        "named_cip_correct": actual_correct,
                        "parity_cosines": similarities[:, index].tolist(),
                        "detection_errors": int((inferred_correct != actual_correct).sum()),
                        "rotation_decision_changes": int((inferred_correct != inferred_correct[0]).sum()),
                        "prediction_sha256": prediction_hash,
                    })
            if digest(prediction_path) != prediction_hash:
                raise ValueError("Prediction changed during diagnosis")
        features["ref_pos"] = positions
        print(json.dumps({"id": sid, "prediction_conditions": len(candidates), "centers": len(centers)}), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["checkpoint"], row["bfloat16_autocast"]].append(row)
    summaries = []
    for (checkpoint, mixed), observations in sorted(groups.items()):
        summaries.append({
            "checkpoint": checkpoint, "bfloat16_autocast": mixed,
            "systems": len({row["id"] for row in observations}), "centers": len(observations),
            "named_cip_correct": sum(row["named_cip_correct"] for row in observations),
            "named_cip_incorrect": sum(not row["named_cip_correct"] for row in observations),
            "centers_with_detection_errors": sum(row["detection_errors"] > 0 for row in observations),
            "detection_errors": sum(row["detection_errors"] for row in observations),
            "observations": 3 * len(observations),
            "rotation_decision_changes": sum(row["rotation_decision_changes"] for row in observations),
            "failed_centers": [row for row in observations if row["detection_errors"]],
        })
    report = {
        "scope": "Read-only detection of named-center stereochemistry errors in saved development predictions; not a repair experiment or an acceptance result.",
        "decision_rule": "Positive dot product between reference and prediction parity vectors means matching handedness. Zero threshold is fixed before evaluation. All 56 components are used; no component selection or fitted classifier is used.",
        "coordinate_handling": "Only diagnostic copies of ligand coordinates are rigidly centered and rotated. Saved prediction coordinates and checkpoint files are never modified.",
        "label_scope": "Modern CIP labels with the original named-atom correspondence. The benchmark instead optimizes graph isomorphism, so symmetric-center attribution can differ.",
        "source_sha256": digest(Path(__file__)),
        "program_sha256": digest(Path(__file__).with_name("program_parity_vector.py")),
        "programmed_module_sha256": state_digest.hexdigest(),
        "plan_sha256": digest(args.plan), "directions": directions, "head_triples": triples,
        "summaries": summaries, "centers": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summaries": [{key: value for key, value in row.items() if key != "failed_centers"} for row in summaries]}))


if __name__ == "__main__":
    main()
