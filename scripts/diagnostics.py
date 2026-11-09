import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import prediction_molecule, read_pdb
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {r["id"]: r for r in manifest["records"]}
    rows = []
    for task in json.loads(args.plan.read_text())["tasks"]:
        root = Path(task["output"])
        file = root / "diagnostics.npz"
        if not file.exists():
            continue
        data = np.load(file)
        run = json.loads((root / "run.json").read_text())
        if run["status"] != "complete":
            continue
        row = records[task["id"]]
        atoms = [a for a in read_pdb(root / run["prediction"]) if a["chain"] == "B"]
        assert len(atoms) == len(data["ref_pos"])
        reference = Chem.MolFromSmiles(row["smiles"])
        from evaluate import chirality
        from rgi_benchmark import stereo_matches

        frames = []
        for label, xyz in [
            ("reference", data["ref_pos"]),
            *[(str(step), xyz) for step, xyz in zip(data["steps"], data["x0"], strict=True)],
        ]:
            predicted_atoms = [{**a, "xyz": coord} for a, coord in zip(atoms, xyz, strict=True)]
            mol = prediction_molecule(row["smiles"], predicted_atoms)
            ez = stereo_matches(mol, mol.GetConformer().GetPositions())[1]
            frames.append({"step": label, **chirality(reference, mol), "ez_all": ez})
        rows.append(
            {
                "id": task["id"],
                "checkpoint": task["checkpoint"],
                "norms_ref_q_total_q_total_c": data["component_norms"].tolist(),
                "frames": frames,
            }
        )
    output = args.plan.with_suffix(".diagnostics.json")
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
