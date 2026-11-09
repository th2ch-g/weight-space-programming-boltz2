import json
import pickle
from pathlib import Path

import torch
from checkpoint_io import load_checkpoint
from rdkit import Chem
from rgi_benchmark import stereo_matches

names = (
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK".split()
)
hits = []
for name in names:
    file = Path(".cache/boltz/mols") / f"{name}.pkl"
    if not file.exists():
        continue
    with file.open("rb") as handle:
        mol = pickle.load(handle)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        label = atom.GetProp("name")
        if len(label) > 1 and label[1].isdigit():
            hits.append((name, label))
assert not hits
manifest = json.loads(Path("data/ct209/manifest.json").read_text())
overflow = []
for row in manifest["records"]:
    file = Path("data/ct209/local_mols") / f"{row['id']}.pkl"
    if file.exists():
        with file.open("rb") as handle:
            mol = pickle.load(handle)
        reference = Chem.MolFromSmiles(row["smiles"])
        match = mol.GetSubstructMatch(reference)
        assert len(match) == reference.GetNumAtoms()
        assert stereo_matches(reference, mol.GetConformer().GetPositions()[list(match)])[1]
        overflow.append({"id": row["id"], "reference_ez_valid": True})
base = load_checkpoint("boltz2_base.ckpt", map_location="cpu", mmap=True)[
    "state_dict"
]
key = "input_embedder.atom_encoder.embed_atom_features.weight"
matrix = base[key].double()
basis = matrix[:, :3]
q, _ = torch.linalg.qr(basis, mode="reduced")
projection = q @ (q.T @ matrix[:, 212:222])
result = {
    "standard_protein_heavy_atoms_using_second_character_digits": hits,
    "ligand_digit_embedding_norm": float(matrix[:, 212:222].norm()),
    "removed_coordinate_subspace_component_norm": float(projection.norm()),
    "overflow_reference_checks": overflow,
}
Path("artifacts/feature_scope_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
