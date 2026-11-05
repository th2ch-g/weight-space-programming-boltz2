import json
import pickle
from pathlib import Path

from boltz.data.parse.schema import compute_3d_conformer
from chemistry import atom_names
from evaluate import chirality
from rdkit import Chem

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
manifest = json.loads(Path("data/ct209/manifest.json").read_text())
directory = Path("data/ct209/local_mols")
directory.mkdir(exist_ok=True)
report = []
for row in manifest["records"]:
    mol = Chem.AddHs(Chem.MolFromSmiles(row["smiles"]))
    ranks = list(Chem.CanonicalRankAtoms(mol))
    historical = [
        a.GetSymbol().upper() + str(rank + 1) for a, rank in zip(mol.GetAtoms(), ranks, strict=True)
    ]
    names = atom_names(mol)
    overflow = names != historical
    report.append({"id": row["id"], "overflow": overflow, "names_unchanged": not overflow})
    if not overflow:
        continue
    destination = directory / f"{row['id']}.pkl"
    if destination.exists():
        raise FileExistsError(destination)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for atom, name in zip(mol.GetAtoms(), names, strict=True):
        atom.SetProp("name", name)
    assert compute_3d_conformer(mol)
    mol = Chem.RemoveHs(mol, sanitize=False)
    reference = Chem.MolFromSmiles(row["smiles"])
    assert Chem.MolToSmiles(mol, isomericSmiles=True) == Chem.MolToSmiles(
        reference, isomericSmiles=True
    )
    report[-1].update(
        chirality=chirality(reference, mol),
        heavy_atoms=mol.GetNumAtoms(),
        atom_names=[a.GetProp("name") for a in mol.GetAtoms()],
    )
    with destination.open("wb") as handle:
        pickle.dump(mol, handle)
Path("artifacts/input_overflow.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"systems": len(report), "overflows": [r for r in report if r["overflow"]]}))
