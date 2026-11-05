import argparse
import json
import sys
from pathlib import Path

import numpy as np
from evaluate import chirality, full_stereochemistry, prediction_molecule, read_pdb
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--bench", type=Path, default=Path("../impl_rgi/bench-rgi"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.bench))
    from common.chiral import is_ez_equal

    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    by_id = {row["id"]: row for row in manifest["records"]}
    plan = json.loads(args.plan.read_text())
    reports = []
    hashes = {}
    for task in plan["tasks"]:
        root = Path(task["output"])
        run = json.loads((root / "run.json").read_text())
        assert run["status"] == "complete"
        control = (
            task["id"],
            task["checkpoint"],
            run["seed"],
            run.get("reference_reflection", False),
        )
        hashes.setdefault(control, []).append(run["prediction_sha256"])
        if not task.get("trace_layers"):
            continue
        data = np.load(root / "layer_trace.npz")
        smiles = by_id[task["id"]]["smiles"]
        atoms = [atom for atom in read_pdb(root / run["prediction"]) if atom["chain"] == "B"]
        reference = Chem.MolFromSmiles(smiles)
        frames = []
        for step, label, xyz in zip(
            data["steps"], data["labels"], data["coordinates"], strict=True
        ):
            assigned = [{**atom, "xyz": p} for atom, p in zip(atoms, xyz, strict=True)]
            molecule = prediction_molecule(smiles, assigned)
            frames.append(
                {
                    "step": int(step),
                    "layer": str(label),
                    **chirality(reference, molecule),
                    **full_stereochemistry(reference, molecule),
                    "ez_all": bool(is_ez_equal(reference, molecule)),
                }
            )
        final_mask = data["labels"] == "decoder_2_output"
        score_mask = data["labels"] == "score_output"
        max_delta = float(
            np.abs(data["coordinates"][final_mask] - data["coordinates"][score_mask]).max()
        )
        assert max_delta < 1e-4
        token_type = data["mol_type"]
        atom_type = data["atom_to_token"] @ token_type
        active_atom = data["atom_pad_mask"].astype(bool)
        reference_xyz = data["ref_pos"][(atom_type == 3) & active_atom]
        reference_atoms = [
            {**atom, "xyz": xyz} for atom, xyz in zip(atoms, reference_xyz, strict=True)
        ]
        encoded_reference = prediction_molecule(smiles, reference_atoms)
        norms = {}
        for name in ("s_inputs", "s_trunk", "conditioning_q", "conditioning_c"):
            types = token_type if name in ("s_inputs", "s_trunk") else atom_type
            active = np.ones(len(types), dtype=bool) if name.startswith("s_") else active_atom
            for kind, value in (("protein", 0), ("ligand", 3)):
                selected = data[name][(types == value) & active]
                norms[f"{name}_{kind}_mean_norm"] = float(np.linalg.norm(selected, axis=-1).mean())
        reports.append(
            {
                "id": task["id"],
                "checkpoint": task["checkpoint"],
                "readout_max_difference": max_delta,
                "encoded_reference": {
                    **chirality(reference, encoded_reference),
                    **full_stereochemistry(reference, encoded_reference),
                },
                "norms": norms,
                "frames": frames,
            }
        )
        print(task["id"], json.dumps([f for f in frames if f["step"] == 199]))
    controls = {
        ":".join(map(str, key)): len(set(values)) == 1
        for key, values in hashes.items()
        if len(values) > 1
    }
    assert all(controls.values()), controls
    report = {
        "interpretation": "Intermediate readout is observational and does not establish causal attribution.",
        "tracing_preserves_prediction_controls": controls,
        "systems": reports,
    }
    args.plan.with_suffix(".layers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"controls": controls, "systems": len(reports)}))


if __name__ == "__main__":
    main()
