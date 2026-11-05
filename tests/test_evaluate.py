import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import (
    chirality,
    full_stereochemistry,
    geometry,
    main,
    named_molecule,
    paired_displacement,
    prediction_molecule,
)


def test_counterfactual_plan_is_excluded_from_benchmark(tmp_path, monkeypatch):
    plan = tmp_path / "diagnostic.json"
    plan.write_text(json.dumps({"stage": "diagnostic", "tasks": []}))
    monkeypatch.setattr(sys, "argv", ["evaluate", str(plan)])
    with pytest.raises(ValueError, match="cannot enter benchmark scoring"):
        main()


def molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 11
    assert AllChem.EmbedMolecule(mol, params) == 0
    return Chem.RemoveHs(mol)


def test_mirror_is_a_chirality_error():
    mol = molecule()
    reference = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    assert chirality(reference, mol)["chirality_all"]
    xyz = mol.GetConformer().GetPositions()
    xyz[:, 0] *= -1
    for i, position in enumerate(xyz):
        mol.GetConformer().SetAtomPosition(i, position)
    assert not chirality(reference, mol)["chirality_all"]


def test_atom_permutation_is_not_a_chirality_error():
    mol = molecule()
    order = list(reversed(range(mol.GetNumAtoms())))
    permuted = Chem.RenumberAtoms(mol, order)
    assert chirality(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"), permuted)["chirality_all"]


def test_ring_ez_is_evaluated():
    reference = Chem.MolFromSmiles("C1CCCCC/C=C/CCCCC1")
    bond = next(b for b in reference.GetBonds() if b.GetStereo() == Chem.BondStereo.STEREOE)
    left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    outer_left, outer_right = bond.GetStereoAtoms()
    predicted = Chem.Mol(reference)
    conformer = Chem.Conformer(predicted.GetNumAtoms())
    for index in range(predicted.GetNumAtoms()):
        conformer.SetAtomPosition(index, (10.0 * index, 2.0, 0.0))
    for index, xyz in (
        (left, (0.0, 0.0, 0.0)),
        (right, (1.34, 0.0, 0.0)),
        (outer_left, (-0.5, 1.0, 0.0)),
        (outer_right, (1.84, -1.0, 0.0)),
    ):
        conformer.SetAtomPosition(index, xyz)
    predicted.AddConformer(conformer)
    result = full_stereochemistry(reference, predicted)
    assert result["defined_ring_ez_bonds"] == 1
    assert result["full_ez_all"]
    predicted.GetConformer().SetAtomPosition(outer_right, (1.84, 1.0, 0.0))
    assert not full_stereochemistry(reference, predicted)["full_ez_all"]


def test_mirror_preserves_ez_but_fails_joint_stereo():
    reference = Chem.MolFromSmiles("N[C@@H](C)/C=C/F")
    predicted = Chem.AddHs(reference)
    params = AllChem.ETKDGv3()
    params.randomSeed = 11
    assert AllChem.EmbedMolecule(predicted, params) == 0
    predicted = Chem.RemoveHs(predicted)
    assert full_stereochemistry(reference, predicted)["full_stereo_all"]
    xyz = predicted.GetConformer().GetPositions()
    xyz[:, 0] *= -1
    for index, position in enumerate(xyz):
        predicted.GetConformer().SetAtomPosition(index, position)
    result = full_stereochemistry(reference, predicted)
    assert result["full_ez_all"]
    assert not result["full_stereo_all"]


def test_name_mapping_is_independent_of_pdb_order():
    smiles = "N[C@@H](C)C(=O)O"
    mol = named_molecule(smiles)
    xyz = molecule().GetConformer().GetPositions()
    atoms = [
        {"name": a.GetProp("name"), "chain": "B", "element": a.GetSymbol(), "xyz": xyz[a.GetIdx()]}
        for a in mol.GetAtoms()
    ]
    observed = prediction_molecule(smiles, list(reversed(atoms)))
    np.testing.assert_array_equal(observed.GetConformer().GetPositions(), xyz)


def test_geometry_detects_stretched_bond():
    mol = molecule()
    xyz = mol.GetConformer().GetPositions()
    assert geometry(mol, xyz)["bond_rms"] == 0
    changed = xyz[0] + np.array([2.0, 0.0, 0.0])
    mol.GetConformer().SetAtomPosition(0, changed)
    assert geometry(mol, xyz)["bond_rms"] > 0.1


def test_paired_pose_removes_only_protein_rigid_motion(tmp_path):
    smiles = "N[C@@H](C)C(=O)O"
    mol = named_molecule(smiles)
    xyz = molecule().GetConformer().GetPositions()
    atoms = [("CA", "A", i + 1, "C", coord) for i, coord in enumerate(np.eye(3))]
    atoms += [(a.GetProp("name"), "B", 1, a.GetSymbol(), xyz[a.GetIdx()]) for a in mol.GetAtoms()]

    def write(path, rotate=False, ligand_shift=0.0):
        lines = []
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        for serial, (name, chain, residue, element, coord) in enumerate(atoms, 1):
            pos = np.array(coord)
            if rotate:
                pos = pos @ rotation + np.array([10.0, 20.0, 30.0])
            if chain == "B":
                pos += np.array([ligand_shift, 0.0, 0.0])
            lines.append(
                f"ATOM  {serial:5d} {name:>4} LIG {chain}{residue:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}\n"
            )
        path.write_text("".join(lines))

    base, rotated, shifted = (
        tmp_path / name for name in ("base.pdb", "rotated.pdb", "shifted.pdb")
    )
    write(base)
    write(rotated, rotate=True)
    write(shifted, ligand_shift=2.0)
    result = paired_displacement({"smiles": smiles}, base, rotated)
    assert result["paired_protein_ca_rmsd"] < 1e-10
    assert result["paired_ligand_rmsd_after_protein_fit"] < 1e-10
    result = paired_displacement({"smiles": smiles}, base, shifted)
    assert abs(result["paired_ligand_rmsd_after_protein_fit"] - 2.0) < 1e-10
