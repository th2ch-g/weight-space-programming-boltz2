import sys
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chemistry import atom_names


def test_preserves_normal_atom_names():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCl"))
    expected = [
        atom.GetSymbol().upper() + str(rank + 1)
        for atom, rank in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol), strict=True)
    ]
    assert atom_names(mol) == expected


def test_overflow_has_unique_short_names_without_changing_graph():
    mol = Chem.AddHs(Chem.MolFromSmiles("Cl" + "C" * 100))
    before = Chem.MolToSmiles(mol)
    historical = [
        atom.GetSymbol().upper() + str(rank + 1)
        for atom, rank in zip(mol.GetAtoms(), Chem.CanonicalRankAtoms(mol), strict=True)
    ]
    assert max(map(len, historical)) > 4
    names = atom_names(mol)
    assert len(names) == mol.GetNumAtoms()
    assert len(set(names)) == len(names)
    assert max(map(len, names)) <= 4
    assert Chem.MolToSmiles(mol) == before
