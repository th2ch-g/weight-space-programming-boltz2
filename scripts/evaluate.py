import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from chemistry import atom_names
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdCIPLabeler
from rgi_benchmark import stereo_matches

RDLogger.DisableLog("rdApp.*")


def named_molecule(smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for atom, name in zip(molecule.GetAtoms(), atom_names(molecule), strict=True):
        atom.SetProp("name", name)
    return Chem.RemoveHs(molecule)


def read_pdb(path):
    atoms = []
    for line in path.read_text().splitlines():
        if line[:6] not in {"ATOM  ", "HETATM"}:
            continue
        if line[16] not in {" ", "A"}:
            continue
        element = line[76:78].strip().title()
        if element == "H":
            continue
        atoms.append(
            {
                "name": line[12:16].strip(),
                "chain": line[21],
                "residue": line[22:27].strip(),
                "element": element,
                "xyz": [float(line[a : a + 8]) for a in (30, 38, 46)],
            }
        )
    return atoms


def prediction_molecule(smiles, atoms):
    mol = named_molecule(smiles)
    ligand = {a["name"]: a for a in atoms if a["chain"] == "B"}
    if len(ligand) != mol.GetNumAtoms():
        raise ValueError("Ligand atom count mismatch")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        item = ligand[atom.GetProp("name")]
        if item["element"] != atom.GetSymbol():
            raise ValueError("Ligand atom element mismatch")
        conf.SetAtomPosition(atom.GetIdx(), tuple(float(x) for x in item["xyz"]))
    mol.AddConformer(conf)
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError("Nonfinite coordinates")
    return mol


def cip_labels(mol, from_3d=False):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    for bond in mol.GetBonds():
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
    if from_3d:
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    return mol, {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}


def chirality(reference, predicted):
    ref, labels = cip_labels(reference)
    pred, observed = cip_labels(predicted, from_3d=True)
    mappings = pred.GetSubstructMatches(ref, useChirality=False, uniquify=False, maxMatches=100000)
    if not mappings:
        raise ValueError("No graph isomorphism")
    correct = max(
        sum(observed.get(mapping[i]) == code for i, code in labels.items()) for mapping in mappings
    )
    return {
        "chiral_centers": len(labels),
        "chiral_correct": correct,
        "chirality_all": correct == len(labels),
    }


def ez_decidable(bond):
    left, right = bond.GetBeginAtom(), bond.GetEndAtom()
    for atom, other in ((left, right), (right, left)):
        neighbors = [
            a for a in atom.GetNeighbors() if a.GetIdx() != other.GetIdx() and a.GetAtomicNum() > 1
        ]
        if not neighbors:
            return False
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0:
            return False
        if any(a.GetAtomicNum() in (7, 8, 16) and a.GetTotalNumHs() > 0 for a in neighbors):
            return False
    return True


def full_stereochemistry(reference, predicted):
    ref, atom_labels = cip_labels(reference)
    pred, observed_atoms = cip_labels(predicted, from_3d=True)
    bond_labels = {}
    unverifiable = 0
    ring_bonds = 0
    for bond in ref.GetBonds():
        if not bond.HasProp("_CIPCode") or bond.GetProp("_CIPCode") not in ("E", "Z"):
            continue
        if not ez_decidable(bond):
            unverifiable += 1
            continue
        key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        bond_labels[key] = bond.GetProp("_CIPCode")
        ring_bonds += int(bond.IsInRing())
    observed_bonds = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))): bond.GetProp("_CIPCode")
        for bond in pred.GetBonds()
        if bond.HasProp("_CIPCode")
    }
    mappings = pred.GetSubstructMatches(ref, useChirality=False, uniquify=False, maxMatches=100000)
    if not mappings:
        raise ValueError("No graph isomorphism for complete stereochemistry evaluation")
    best_bonds = 0
    joint = False
    for mapping in mappings:
        atom_correct = all(
            observed_atoms.get(mapping[i]) == code for i, code in atom_labels.items()
        )
        bond_correct = sum(
            observed_bonds.get(tuple(sorted((mapping[i], mapping[j])))) == code
            for (i, j), code in bond_labels.items()
        )
        best_bonds = max(best_bonds, bond_correct)
        joint |= atom_correct and bond_correct == len(bond_labels)
    return {
        "full_ez_bonds": len(bond_labels),
        "full_ez_correct_bonds": best_bonds,
        "full_ez_all": best_bonds == len(bond_labels) if bond_labels else None,
        "full_stereo_all": joint,
        "defined_ring_ez_bonds": ring_bonds,
        "unverifiable_ez_bonds": unverifiable,
    }


def angle(x, i, j, k):
    a, b = x[i] - x[j], x[k] - x[j]
    return np.degrees(
        np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))
    )


def geometry(mol, reference_xyz):
    xyz = mol.GetConformer().GetPositions()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    bond_errors = [
        np.linalg.norm(xyz[i] - xyz[j]) - np.linalg.norm(reference_xyz[i] - reference_xyz[j])
        for i, j in bonds
    ]
    angle_errors = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        for i, k in itertools.combinations([a.GetIdx() for a in atom.GetNeighbors()], 2):
            angle_errors.append(angle(xyz, i, j, k) - angle(reference_xyz, i, j, k))
    return {
        "bond_rms": float(np.sqrt(np.mean(np.square(bond_errors)))),
        "angle_rms": float(np.sqrt(np.mean(np.square(angle_errors)))),
    }


def ideal(smiles, directory):
    key = hashlib.sha256(smiles.encode()).hexdigest()
    path = directory / f"{key}.npy"
    if path.exists():
        return np.load(path)
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20261103
    params.numThreads = 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("Independent reference embedding failed")
    status = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    if status != 0:
        raise ValueError("Independent reference UFF did not converge")
    mol = Chem.RemoveHs(mol)
    if not chirality(Chem.MolFromSmiles(smiles), mol)["chirality_all"]:
        raise ValueError("Independent reference chirality mismatch")
    xyz = mol.GetConformer().GetPositions()
    directory.mkdir(parents=True, exist_ok=True)
    np.save(path, xyz)
    return xyz


def clashes(mol, atoms):
    table = Chem.GetPeriodicTable()
    xyz = mol.GetConformer().GetPositions()
    radii = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    protein = [a for a in atoms if a["chain"] != "B"]
    pxyz = np.array([a["xyz"] for a in protein])
    prad = np.array([table.GetRvdw(table.GetAtomicNumber(a["element"])) for a in protein])
    relative = np.linalg.norm(xyz[:, None] - pxyz[None], axis=-1) / (radii[:, None] + prad[None])
    topology = Chem.GetDistanceMatrix(mol)
    i, j = np.where(np.triu(topology > 2, 1))
    intra = np.linalg.norm(xyz[i] - xyz[j], axis=-1) / (radii[i] + radii[j])
    return {
        "inter_clash_min_relative": float(relative.min()),
        "inter_clash_pairs": int((relative < 0.75).sum()),
        "intra_clash_min_relative": float(intra.min()) if len(intra) else None,
        "intra_clash_pairs": int((intra < 0.75).sum()),
    }


def paired_displacement(row, base_path, edited_path):
    base_atoms = read_pdb(base_path)
    edited_atoms = read_pdb(edited_path)
    base_ca = {
        (a["chain"], a["residue"]): a["xyz"]
        for a in base_atoms
        if a["chain"] != "B" and a["name"] == "CA"
    }
    edited_ca = {
        (a["chain"], a["residue"]): a["xyz"]
        for a in edited_atoms
        if a["chain"] != "B" and a["name"] == "CA"
    }
    if base_ca.keys() != edited_ca.keys() or len(base_ca) < 3:
        raise ValueError("Protein CA correspondence mismatch")
    a = np.array(list(base_ca.values()))
    b = np.array([edited_ca[key] for key in base_ca])
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - bc).T @ (a - ac))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    transformed = (b - bc) @ rotation + ac
    reference = prediction_molecule(row["smiles"], base_atoms)
    predicted = prediction_molecule(row["smiles"], edited_atoms)
    x = reference.GetConformer().GetPositions()
    y = (predicted.GetConformer().GetPositions() - bc) @ rotation + ac
    mappings = predicted.GetSubstructMatches(
        reference, uniquify=False, useChirality=False, maxMatches=100000
    )
    ligand_rms = min(
        float(np.sqrt(np.mean(np.sum((x - y[list(mapping)]) ** 2, axis=-1))))
        for mapping in mappings
    )
    return {
        "paired_protein_ca_rmsd": float(np.sqrt(np.mean(np.sum((a - transformed) ** 2, axis=-1)))),
        "paired_ligand_rmsd_after_protein_fit": ligand_rms,
        "paired_gpu_equal": None,
    }


def evaluate(row, path, reference_dir):
    atoms = read_pdb(path)
    mol = prediction_molecule(row["smiles"], atoms)
    reference = Chem.MolFromSmiles(row["smiles"])
    benchmark_chirality, benchmark_ez = stereo_matches(mol, mol.GetConformer().GetPositions())
    result = {
        **chirality(reference, mol),
        **full_stereochemistry(reference, mol),
        "benchmark_chirality_all": benchmark_chirality,
        "ez_all": benchmark_ez,
        **clashes(mol, atoms),
    }
    result["stereo_all"] = result["chirality_all"] and result["ez_all"] is not False
    try:
        result.update(
            {"ideal_" + k: v for k, v in geometry(mol, ideal(row["smiles"], reference_dir)).items()}
        )
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        result["ideal_error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("stage") == "diagnostic":
        raise ValueError("Counterfactual diagnostic plans cannot enter benchmark scoring")
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {r["id"]: r for r in manifest["records"]}
    scores = []
    for task in plan["tasks"]:
        runfile = Path(task["output"]) / "run.json"
        result = {"id": task["id"], "checkpoint": task["checkpoint"], "output": task["output"]}
        if not runfile.exists():
            result["status"] = "missing"
        else:
            run = json.loads(runfile.read_text())
            result["status"] = run["status"]
            if run.get("reference_reflection"):
                result["status"] = "diagnostic_excluded"
                scores.append(result)
                continue
            if run["status"] == "complete":
                try:
                    result.update(
                        evaluate(
                            records[task["id"]],
                            runfile.parent / run["prediction"],
                            Path("artifacts/ideal"),
                        )
                    )
                except (KeyError, ValueError, RuntimeError, OSError) as exc:
                    result.update(status="evaluation_failed", error=str(exc))
        scores.append(result)
    out = args.plan.with_suffix(".scores.json")
    base_tasks = {t["id"]: t for t in plan["tasks"] if t["checkpoint"] == "boltz2_base.ckpt"}
    for score in scores:
        if score["status"] != "complete" or score["checkpoint"] == "boltz2_base.ckpt":
            continue
        if score["id"] not in base_tasks:
            continue
        base_dir = Path(base_tasks[score["id"]]["output"])
        base_run = json.loads((base_dir / "run.json").read_text())
        run_dir = Path(score["output"])
        edited_run = json.loads((run_dir / "run.json").read_text())
        if base_run["status"] != "complete":
            continue
        score.update(
            paired_displacement(
                records[score["id"]],
                base_dir / base_run["prediction"],
                run_dir / edited_run["prediction"],
            )
        )
        score["paired_gpu_equal"] = base_run["gpu"] == edited_run["gpu"]
    out.write_text(json.dumps(scores, indent=2) + "\n")
    for ckpt in sorted({s["checkpoint"] for s in scores}):
        rows = [s for s in scores if s["checkpoint"] == ckpt]
        complete = [s for s in rows if s["status"] == "complete"]
        summary = {"checkpoint": ckpt, "planned": len(rows), "complete": len(complete)}
        for key in (
            "chirality_all",
            "ez_all",
            "stereo_all",
            "full_ez_all",
            "full_stereo_all",
            "ideal_bond_rms",
            "ideal_angle_rms",
            "inter_clash_min_relative",
            "paired_protein_ca_rmsd",
            "paired_ligand_rmsd_after_protein_fit",
        ):
            values = [r[key] for r in complete if r.get(key) is not None]
            summary[key] = (
                {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
                if values
                else None
            )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
