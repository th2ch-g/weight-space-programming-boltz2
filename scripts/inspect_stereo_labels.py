import json
from pathlib import Path

from evaluate import cip_labels, ez_decidable
from rdkit import Chem


def main():
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    systems = []
    for record in manifest["records"]:
        molecule, labels = cip_labels(Chem.MolFromSmiles(record["smiles"]))
        bonds = [
            bond
            for bond in molecule.GetBonds()
            if bond.HasProp("_CIPCode") and bond.GetProp("_CIPCode") in ("E", "Z")
        ]
        systems.append(
            {
                "id": record["id"],
                "split": record["split"],
                "defined_chiral_centers": len(labels),
                "defined_ez_bonds": len(bonds),
                "unverifiable_ez_bonds": [bond.GetIdx() for bond in bonds if not ez_decidable(bond)],
            }
        )
    report = {
        "scope": "Input labels only; no test predictions or test coordinates are inspected.",
        "systems": systems,
        "without_chirality": [row["id"] for row in systems if not row["defined_chiral_centers"]],
        "without_ez": [row["id"] for row in systems if not row["defined_ez_bonds"]],
        "with_unverifiable_ez": [row for row in systems if row["unverifiable_ez_bonds"]],
    }
    Path("artifacts/stereo_label_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "systems"}))


if __name__ == "__main__":
    main()
