import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


manifest = json.loads(Path("data/ct209/manifest.json").read_text())
groups = {}
scaffolds = {}
sequences = {}
for row in manifest["records"]:
    sid = row["id"]
    assert digest(Path(row["input"])) == row["input_sha256"]
    assert digest(Path("data/ct209/msa") / f"{sid}.a3m") == row["msa_sha256"]
    assert digest(Path("data/ct209/sdf") / f"{sid}.sdf") == row["sdf_sha256"]
    doc = yaml.safe_load(Path(row["input"]).read_text())
    assert set(doc) == {"sequences", "version"}
    ligand = next(x["ligand"] for x in doc["sequences"] if "ligand" in x)
    assert set(ligand) == {"id", "smiles"}
    assert ligand["smiles"] == row["smiles"]
    mol = Chem.MolFromSmiles(row["smiles"])
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    scaffold = scaffold or Chem.MolToSmiles(mol, isomericSmiles=False)
    assert scaffold == row["scaffold"]
    for mapping, key in (
        (groups, row["group"]),
        (scaffolds, scaffold),
        (sequences, row["sequence"]),
    ):
        assert mapping.setdefault(key, row["split"]) == row["split"]
result = {
    "systems": len(manifest["records"]),
    "all_input_hashes_match": True,
    "guidance_inputs_absent": True,
    "group_leakage": False,
    "scaffold_leakage": False,
    "exact_sequence_leakage": False,
    "scaffolds_reproduced": True,
    "dedicated_environment_versions": {
        name: version(name) for name in ("boltz", "torch", "rdkit", "biopython")
    },
}
Path("artifacts/dataset_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
