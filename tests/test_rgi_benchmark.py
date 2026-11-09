import os
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rgi_benchmark import stereo_matches

pytestmark = pytest.mark.skipif(
    not os.environ.get("RGI_BENCHMARK"), reason="RGI_BENCHMARK is not set"
)


def coordinates(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return Chem.RemoveHs(mol).GetConformer().GetPositions()


def test_public_benchmark_detects_reflection_and_double_bond_isomer():
    reference = Chem.MolFromSmiles("C/C=C/[C@H](O)CC")
    xyz = coordinates("C/C=C/[C@H](O)CC")
    assert stereo_matches(reference, xyz) == (True, True)
    assert stereo_matches(reference, xyz * np.array([-1.0, 1.0, 1.0])) == (False, True)
    assert stereo_matches(reference, coordinates("C/C=C\\[C@H](O)CC")) == (True, False)


def test_missing_sites_are_not_scored():
    assert stereo_matches(Chem.MolFromSmiles("CCO"), coordinates("CCO")) == (None, None)
