import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate import cip_labels, named_molecule, prediction_molecule, read_pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--trace-plan", type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {row["id"]: row for row in manifest["records"]}
    atom_windows = {}
    if args.trace_plan:
        dimensions = json.loads(Path("artifacts/weight_inspection.json").read_text())["hyper_parameters"]
        query_width = int(dimensions["atoms_per_window_queries"])
        key_width = int(dimensions["atoms_per_window_keys"])
        for task in json.loads(args.trace_plan.read_text())["tasks"]:
            if task["id"] not in args.ids or not task.get("trace_layers"):
                continue
            root = Path(task["output"])
            with np.load(root / "layer_trace.npz") as arrays:
                atom_type = arrays["atom_to_token"] @ arrays["mol_type"]
                indices = np.flatnonzero((atom_type == 3) & (arrays["atom_pad_mask"] > 0))
            run = json.loads((root / "run.json").read_text())
            atoms = [a for a in read_pdb(root / run["prediction"]) if a["chain"] == "B"]
            atom_windows[task["id"]] = {
                atom["name"]: int(index) for atom, index in zip(atoms, indices, strict=True)
            }
    observations = defaultdict(list)
    seen = set()
    for path in args.scores:
        for result in json.loads(path.read_text()):
            if result["id"] not in args.ids or result["status"] != "complete":
                continue
            if result["output"] in seen:
                continue
            seen.add(result["output"])
            row = records[result["id"]]
            if row["split"] != "development":
                raise ValueError("Persistent-center diagnosis is restricted to development")
            root = Path(result["output"])
            run = json.loads((root / "run.json").read_text())
            if run.get("reference_reflection"):
                raise ValueError("Counterfactual inputs cannot enter this comparison")
            predicted = prediction_molecule(row["smiles"], read_pdb(root / run["prediction"]))
            labels = cip_labels(predicted, from_3d=True)[1]
            observations[row["id"]].append((result["checkpoint"], labels))
    systems = []
    for sid in args.ids:
        reference, labels = cip_labels(named_molecule(records[sid]["smiles"]))
        centers = []
        for index, expected in labels.items():
            atom = reference.GetAtomWithIdx(index)
            window = {}
            if sid in atom_windows:
                global_index = atom_windows[sid][atom.GetProp("name")]
                start = (global_index // query_width) * query_width - (key_width - query_width) // 2
                window = {
                    "global_atom_index": global_index,
                    "direct_key_window": [start, start + key_width],
                    "neighbors_outside_direct_window": [
                        neighbor.GetProp("name") for neighbor in atom.GetNeighbors()
                        if not start <= atom_windows[sid][neighbor.GetProp("name")] < start + key_width
                    ],
                }
            correct = sorted(
                {checkpoint for checkpoint, observed in observations[sid] if observed.get(index) == expected}
            )
            centers.append(
                {
                    "atom_index": index,
                    "atom_name": atom.GetProp("name"),
                    "expected_cip": expected,
                    "element": atom.GetSymbol(),
                    "heavy_atom_degree": atom.GetDegree(),
                    "in_ring": atom.IsInRing(),
                    "neighbor_elements": [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()],
                    "ever_correct_checkpoints": correct,
                    "never_correct_in_observed_predictions": not correct,
                    **window,
                }
            )
        systems.append({"id": sid, "predictions": len(observations[sid]), "centers": centers})
    report = {
        "scope": "Descriptive named-atom diagnosis across tested development predictions; not a matched performance comparison.",
        "sources": [path.as_posix() for path in args.scores],
        "systems": systems,
    }
    Path("artifacts/persistent_centers.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in systems:
        print(json.dumps({
            "id": row["id"], "predictions": row["predictions"],
            "persistent_errors": [center for center in row["centers"] if center["never_correct_in_observed_predictions"]],
        }))


if __name__ == "__main__":
    main()
