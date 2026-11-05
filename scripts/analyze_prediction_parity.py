import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import (
    cip_labels,
    full_stereochemistry,
    paired_displacement,
    prediction_molecule,
    read_pdb,
)
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["stage"] != "diagnostic":
        raise ValueError("This analysis requires a diagnostic plan")
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {row["id"]: row for row in manifest["records"]}
    tasks = {
        (task["id"], task["checkpoint"], task["reference_reflection"]): task
        for task in plan["tasks"]
    }
    results = []
    for sid, checkpoint in sorted({(task["id"], task["checkpoint"]) for task in plan["tasks"]}):
        row = records[sid]
        roots = [Path(tasks[sid, checkpoint, flag]["output"]) for flag in (False, True)]
        runs = [json.loads((root / "run.json").read_text()) for root in roots]
        if any(run["status"] != "complete" for run in runs):
            raise ValueError(f"Incomplete diagnostic pair: {sid}, {checkpoint}")
        for key in (
            "input_sha256", "msa_sha256", "settings", "versions", "gpu", "job_id",
            "runner_sha256", "source_sha256", "checkpoint_sha256",
        ):
            if runs[0][key] != runs[1][key]:
                raise ValueError(f"Diagnostic pair mismatch: {key}")
        positions = [np.load(root / "diagnostics.npz")["ref_pos"] for root in roots]
        expected = positions[0].copy()
        expected[:, 0] = 2 * expected[:, 0].mean() - expected[:, 0]
        reflection_error = float(np.max(np.abs(positions[1] - expected)))
        if reflection_error > 1e-4:
            raise ValueError(f"Reference reflection mismatch: {reflection_error}")
        input_molecule, input_labels = cip_labels(Chem.MolFromSmiles(row["smiles"]))
        prediction_labels = []
        reference_labels = []
        ez_results = []
        for root, run, xyz in zip(roots, runs, positions, strict=True):
            atoms = read_pdb(root / run["prediction"])
            predicted = prediction_molecule(row["smiles"], atoms)
            ligand_atoms = [atom for atom in atoms if atom["chain"] == "B"]
            reference = prediction_molecule(
                row["smiles"],
                [{**atom, "xyz": coord} for atom, coord in zip(ligand_atoms, xyz, strict=True)],
            )
            reference_labels.append(cip_labels(reference, from_3d=True)[1])
            prediction_labels.append(cip_labels(predicted, from_3d=True)[1])
            ez_results.append(full_stereochemistry(input_molecule, predicted)["full_ez_all"])
        if any(reference_labels[0].get(index) != code for index, code in input_labels.items()):
            raise ValueError("Normal reference does not preserve input chirality")
        expected_changes = [
            index for index in input_labels
            if reference_labels[0].get(index) != reference_labels[1].get(index)
        ]
        observed_changes = [
            index for index in input_labels
            if prediction_labels[0].get(index) != prediction_labels[1].get(index)
        ]
        results.append(
            {
                "id": sid,
                "checkpoint": checkpoint,
                "defined_centers": len(input_labels),
                "expected_changed_centers": len(expected_changes),
                "observed_changed_centers": len(observed_changes),
                "observed_changed_atom_indices": observed_changes,
                "correct_against_each_reference": [
                    sum(predicted.get(index) == reference.get(index) for index in input_labels)
                    for predicted, reference in zip(prediction_labels, reference_labels, strict=True)
                ],
                "full_ez_all_normal_and_reflected": ez_results,
                "reference_reflection_max_error": reflection_error,
                **paired_displacement(
                    row, roots[0] / runs[0]["prediction"], roots[1] / runs[1]["prediction"]
                ),
            }
        )
    output = {
        "scope": "Counterfactual reference sensitivity; excluded from repair acceptance.",
        "interpretation": "Center comparisons use fixed named-atom correspondence. Reflection preserves reference distances, bond angles, and E/Z while reversing handedness. Changed output labels alone do not establish correct transfer.",
        "systems": results,
    }
    args.plan.with_suffix(".analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
