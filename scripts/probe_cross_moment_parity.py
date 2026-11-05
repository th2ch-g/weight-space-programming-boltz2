import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate import cip_labels, named_molecule, prediction_molecule, read_pdb


def cross_moment(reference, predicted, center_weight):
    reference = np.column_stack((reference, np.ones(len(reference))))
    predicted = np.column_stack((predicted, np.ones(len(predicted))))
    weights = np.ones(len(reference))
    weights[0] = center_weight
    moment = reference.T @ (weights[:, None] * predicted)
    return float(np.linalg.det(moment))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {row["id"]: row for row in json.loads(Path("data/ct209/manifest.json").read_text())["records"]}
    observations = []
    sources = {}
    generator = np.random.default_rng(20261105)
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    for plan_path in args.plans:
        sources[plan_path.as_posix()] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        for task in json.loads(plan_path.read_text())["tasks"]:
            if records[task["id"]]["split"] != "development":
                raise ValueError("Only development predictions may be inspected")
            root = Path(task["output"])
            run = json.loads((root / "run.json").read_text())
            if run["status"] != "complete" or run.get("diagnostic") or run.get("reference_reflection"):
                raise ValueError("Only ordinary completed predictions may be inspected")
            pdb_path = root / run["prediction"]
            if hashlib.sha256(pdb_path.read_bytes()).hexdigest() != run["prediction_sha256"]:
                raise ValueError("Prediction hash mismatch")
            trace_path = root / "layer_trace.npz"
            trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            with np.load(trace_path) as trace:
                atom_types = trace["atom_to_token"] @ trace["mol_type"]
                mask = (atom_types == 3) & trace["atom_pad_mask"].astype(bool)
                reference = trace["ref_pos"][mask].astype(np.float64)
            atoms = read_pdb(pdb_path)
            ligand = [atom for atom in atoms if atom["chain"] == "B"]
            by_name = {atom["name"]: index for index, atom in enumerate(ligand)}
            if len(by_name) != len(reference):
                raise ValueError("Ligand atom mapping mismatch")
            predicted = np.array([atom["xyz"] for atom in ligand], dtype=np.float64)
            molecule, expected = cip_labels(named_molecule(records[task["id"]]["smiles"]))
            observed_mol, observed = cip_labels(prediction_molecule(records[task["id"]]["smiles"], atoms), from_3d=True)
            observed_names = {observed_mol.GetAtomWithIdx(index).GetProp("name"): value for index, value in observed.items()}
            for center, expected_cip in expected.items():
                atom = molecule.GetAtomWithIdx(center)
                name = atom.GetProp("name")
                indices = [by_name[name]] + [by_name[neighbor.GetProp("name")] for neighbor in atom.GetNeighbors()]
                if len(indices) not in (4, 5):
                    raise ValueError("Expected three or four heavy neighbors")
                ref = reference[indices] - reference[indices].mean(0)
                pred = predicted[indices] - predicted[indices].mean(0)
                correct = observed_names.get(name) == expected_cip
                for center_weight in (1.0, 1 / 256):
                    determinant = cross_moment(ref, pred, center_weight)
                    rotated = cross_moment(ref @ rotation + [2, -1, 3], pred, center_weight)
                    reflected = cross_moment(-ref @ rotation + [2, -1, 3], pred, center_weight)
                    observations.append({
                        "plan": plan_path.as_posix(), "id": task["id"], "checkpoint": task["checkpoint"],
                        "atom_name": name, "heavy_neighbors": len(indices) - 1,
                        "center_weight": center_weight, "named_cip_correct": correct,
                        "expected_cip": expected_cip, "predicted_cip": observed_names.get(name),
                        "determinant": determinant, "misclassified": bool((determinant > 0) != correct),
                        "proper_rotation_sign_error": bool(np.sign(rotated) != np.sign(determinant)),
                        "reflection_sign_error": bool(np.sign(reflected) != -np.sign(determinant)),
                        "prediction_sha256": run["prediction_sha256"],
                        "trace_sha256": trace_hash,
                    })
    groups = defaultdict(list)
    for row in observations:
        groups[row["plan"], row["checkpoint"], row["center_weight"]].append(row)
    summaries = []
    for (plan, checkpoint, weight), rows in sorted(groups.items()):
        summaries.append({
            "plan": plan, "checkpoint": checkpoint, "center_weight": weight,
            "centers": len(rows), "named_cip_correct": sum(row["named_cip_correct"] for row in rows),
            "misclassified": sum(row["misclassified"] for row in rows),
            "proper_rotation_sign_errors": sum(row["proper_rotation_sign_error"] for row in rows),
            "reflection_sign_errors": sum(row["reflection_sign_error"] for row in rows),
            "failed_centers": [row for row in rows if row["misclassified"]],
        })
    report = {
        "scope": "Read-only float64 algebra diagnostic on saved development coordinates. Not a Boltz weight program or a chemistry repair.",
        "formula": "det(sum_i w_i [r_i,1]^T [p_i,1]); neighbors have unit weight and the center has fixed weight 1 or 1/256.",
        "reasoning": "Cauchy-Binet expresses this determinant as a sum of products of oriented tetrahedron volumes. Three heavy neighbors give one term; four heavy neighbors give one neighbor-only term plus four center-weighted terms.",
        "limitations": "The four-neighbor sum is not universally equivalent to the neighbor-only tetrahedron sign. A fixed small center weight is a diagnostic hypothesis, not a guaranteed classifier. No mixed-precision neural implementation is tested.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_hashes": sources, "summaries": summaries, "observations": observations,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in row.items() if key != "failed_centers"} for row in summaries]))


if __name__ == "__main__":
    main()
