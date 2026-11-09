"""Ligand stereochemistry and geometry metrics from the public RGI-Toolkit benchmark."""

import hashlib
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np
from rdkit import Chem

REPOSITORY = "https://github.com/cddlab/rgi_toolkit"
TAG = "v0.5.0"
COMMIT = "900aff9e73553e3113cba5506c62262f0beb9fdb"
SOURCE_SHA256 = {
    "common.py": "8bb949b2d2f9285bfc2dafdb3f488210b0950de1fc89e0daff81fa5dd283564f",
    "metrics.py": "6c5bfaeb71e37c988826bd17b72c6da6fae2fca8c874c8c187cc61d0d09adb6c",
}


def directory():
    value = os.environ.get("RGI_BENCHMARK")
    if not value:
        raise RuntimeError(
            f"Set RGI_BENCHMARK to the benchmark directory of {REPOSITORY} at {TAG} ({COMMIT})"
        )
    return Path(value).resolve()


def source_hashes(path=None):
    path = path or directory()
    hashes = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in SOURCE_SHA256
    }
    if hashes != SOURCE_SHA256:
        raise ValueError(f"RGI benchmark sources differ from {REPOSITORY} at {COMMIT}")
    return hashes


@cache
def load():
    path = directory()
    source_hashes(path)
    sys.path.insert(0, str(path))
    try:
        import common
        import metrics
    finally:
        sys.path.remove(str(path))
    if any(Path(module.__file__).resolve().parent != path for module in (common, metrics)):
        raise ImportError("Another module shadows the RGI benchmark sources")
    return metrics


def stereo_matches(reference, xyz):
    """Return R/S and E/Z agreement of coordinates with the graph stereochemistry.

    `xyz` follows the heavy-atom order of `reference`. A value is None when the
    reference has no site evaluated by the benchmark.
    """
    metrics = load()
    chiral, ez = metrics._graph_stereo_labels(reference)
    expected = chiral, metrics.benchmark_ez(reference, ez)
    observed = metrics._coordinate_stereo_labels(reference, np.asarray(xyz, dtype=float))
    return tuple(
        all(seen.get(site) == label for site, label in labels.items()) if labels else None
        for labels, seen in zip(expected, observed, strict=True)
    )


def conformer_metrics(mol, crystal_sdf):
    """Benchmark conformer metrics for a SMILES-ordered molecule with predicted coordinates."""
    metrics = load()
    pred = np.asarray(mol.GetConformer().GetPositions())
    reference, ideal = metrics.ideal(Chem.MolToSmiles(mol))
    match = reference.GetSubstructMatch(mol, useChirality=True)
    if len(match) != len(pred):
        raise ValueError("Reference graph mapping failed")
    bond, angle = metrics.geometry_difference(mol, pred, ideal[list(match)])
    chiral, ez = stereo_matches(mol, pred)
    crystal = next(iter(Chem.SDMolSupplier(str(crystal_sdf), removeHs=True)), None)
    if crystal is None:
        raise ValueError("Cannot read crystal ligand SDF")
    match = crystal.GetSubstructMatch(mol, useChirality=True)
    if len(match) != len(pred):
        raise ValueError("Crystal ligand graph differs from the input SMILES")
    crystal_xyz = crystal.GetConformer().GetPositions()[list(match)]
    crystal_bond, crystal_angle = metrics.geometry_difference(mol, pred, crystal_xyz)
    return {
        "chiral_match": chiral, "ez_match": ez,
        "ideal_bond_rms": bond, "ideal_angle_rms": angle,
        "crystal_bond_rms": crystal_bond, "crystal_angle_rms": crystal_angle,
        "ligand_rmsd": metrics.rmsd(pred, crystal_xyz),
    }
