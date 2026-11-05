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


def reflected(positions):
    result = positions.copy()
    result[:, 0] = 2 * result[:, 0].mean() - result[:, 0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["stage"] != "diagnostic":
        raise ValueError("Reference-path analysis requires a diagnostic plan")
    records = {
        row["id"]: row for row in json.loads(Path("data/ct209/manifest.json").read_text())["records"]
    }
    tasks = {(task["id"], task["reflection_path"]): task for task in plan["tasks"]}
    results = []
    for sid in dict.fromkeys(task["id"] for task in plan["tasks"]):
        if records[sid]["split"] != "development":
            raise ValueError("Path diagnosis is restricted to development")
        roots = {mode: Path(tasks[sid, mode]["output"]) for mode in ("none", "input", "denoiser", "all")}
        runs = {mode: json.loads((root / "run.json").read_text()) for mode, root in roots.items()}
        if any(run["status"] != "complete" or not run.get("diagnostic") for run in runs.values()):
            raise ValueError("Each condition must be a completed diagnostic prediction")
        reference_coordinates = {}
        labels = {}
        ez = {}
        expected, expected_labels = cip_labels(Chem.MolFromSmiles(records[sid]["smiles"]))
        normal_atoms = read_pdb(roots["none"] / runs["none"]["prediction"])
        ligand_atoms = [atom for atom in normal_atoms if atom["chain"] == "B"]
        normal_reference_labels = None
        reflected_reference_labels = None
        for mode, root in roots.items():
            for key in (
                "input_sha256", "msa_sha256", "settings", "versions", "gpu", "job_id",
                "runner_sha256", "source_sha256", "checkpoint_sha256", "seed",
            ):
                if runs[mode][key] != runs["none"][key]:
                    raise ValueError(f"Diagnostic condition mismatch: {sid}, {mode}, {key}")
            if runs[mode]["reflection_path"] != mode:
                raise ValueError("Recorded reflection path differs from the planned intervention")
            with np.load(root / "diagnostics.npz") as arrays:
                reference_coordinates[mode] = {
                    name: arrays[name].copy() for name in ("input_ref_pos", "denoiser_ref_pos")
                }
            predicted = prediction_molecule(
                records[sid]["smiles"], read_pdb(root / runs[mode]["prediction"])
            )
            labels[mode] = cip_labels(predicted, from_3d=True)[1]
            ez[mode] = full_stereochemistry(expected, predicted)["full_ez_all"]
        for parity in ("normal", "reflected"):
            positions = reference_coordinates["none"]["input_ref_pos"]
            if parity == "reflected":
                positions = reflected(positions)
            molecule = prediction_molecule(records[sid]["smiles"], [
                {**atom, "xyz": position}
                for atom, position in zip(ligand_atoms, positions, strict=True)
            ])
            current = cip_labels(molecule, from_3d=True)[1]
            if parity == "normal":
                normal_reference_labels = current
            else:
                reflected_reference_labels = current
        if any(normal_reference_labels.get(index) != label for index, label in expected_labels.items()):
            raise ValueError("Normal reference is inconsistent with input stereochemistry")
        for mode in roots:
            errors = {}
            for name, inverted in (
                ("input_ref_pos", mode in ("input", "all")),
                ("denoiser_ref_pos", mode in ("denoiser", "all")),
            ):
                normal = reference_coordinates["none"][name]
                target = reflected(normal) if inverted else normal
                error = float(np.abs(reference_coordinates[mode][name] - target).max())
                if error > 1e-4:
                    raise ValueError(f"Reflection path isolation failed: {sid}, {mode}, {name}, {error}")
                errors[name] = error
            changed = [index for index in expected_labels if labels[mode].get(index) != labels["none"].get(index)]
            results.append({
                "id": sid, "path": mode, "defined_centers": len(expected_labels),
                "reference_path_max_errors": errors,
                "changed_output_centers": len(changed), "changed_atom_indices": changed,
                "output_correct_against_normal_reference": sum(
                    labels[mode].get(index) == normal_reference_labels.get(index) for index in expected_labels
                ),
                "output_correct_against_reflected_reference": sum(
                    labels[mode].get(index) == reflected_reference_labels.get(index) for index in expected_labels
                ),
                "full_ez_all": ez[mode],
            })
            results[-1].update(paired_displacement(
                records[sid], roots["none"] / runs["none"]["prediction"], roots[mode] / runs[mode]["prediction"]
            ))
            print(json.dumps(results[-1]), flush=True)
    args.plan.with_suffix(".analysis.json").write_text(json.dumps({
        "scope": "Counterfactual separation of input/trunk and denoiser reference paths; excluded from repair acceptance.",
        "interpretation": "Changes use named-atom correspondence. Isolating a responsive path does not by itself establish correct stereochemistry transfer or identify a valid weight edit.",
        "systems": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
