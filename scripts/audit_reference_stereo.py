import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate import chirality, cip_labels, full_stereochemistry, prediction_molecule, read_pdb
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {row["id"]: row for row in manifest["records"]}
    hashes = defaultdict(dict)
    systems = []
    for task in plan["tasks"]:
        root = Path(task["output"])
        run = json.loads((root / "run.json").read_text())
        if run["status"] != "complete" or not task.get("trace_layers"):
            raise ValueError("Reference audit requires completed traced predictions")
        if run.get("diagnostic") or run.get("reference_reflection"):
            raise ValueError("Counterfactual references cannot enter this audit")
        with np.load(root / "layer_trace.npz") as trace:
            types = trace["atom_to_token"] @ trace["mol_type"]
            mask = (types == 3) & trace["atom_pad_mask"].astype(bool)
            positions = trace["ref_pos"][mask]
        hashes[task["id"]][task["checkpoint"]] = hashlib.sha256(positions.tobytes()).hexdigest()
        if task["checkpoint"] != "boltz2_base.ckpt":
            continue
        row = records[task["id"]]
        atoms = [atom for atom in read_pdb(root / run["prediction"]) if atom["chain"] == "B"]
        reference_atoms = [
            {**atom, "xyz": position}
            for atom, position in zip(atoms, positions, strict=True)
        ]
        encoded = prediction_molecule(row["smiles"], reference_atoms)
        expected = Chem.MolFromSmiles(row["smiles"])
        labels = cip_labels(expected)[1]
        observed = cip_labels(encoded, from_3d=True)[1]
        systems.append({
            "id": task["id"],
            **chirality(expected, encoded),
            **full_stereochemistry(expected, encoded),
            "named_center_disagreements": [
                {"atom_index": index, "expected": label, "observed": observed.get(index)}
                for index, label in labels.items() if observed.get(index) != label
            ],
        })
    paired_equal = {sid: len(set(values.values())) == 1 for sid, values in hashes.items()}
    report = {
        "scope": "Actual saved ligand reference coordinates supplied to ordinary predictions; no inputs or outputs modified.",
        "plan": args.plan.as_posix(),
        "base_systems": len(systems),
        "all_rs_correct": all(row["chirality_all"] for row in systems),
        "all_ez_correct": all(row["full_ez_all"] is True for row in systems),
        "all_joint_correct": all(row["full_stereo_all"] for row in systems),
        "all_paired_references_identical": all(paired_equal.values()),
        "reference_sha256": dict(hashes),
        "paired_references_equal": paired_equal,
        "systems": systems,
    }
    args.plan.with_suffix(".reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "base_systems", "all_rs_correct", "all_ez_correct", "all_joint_correct", "all_paired_references_identical"
    )}))
    for row in systems:
        if not row["full_stereo_all"]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
