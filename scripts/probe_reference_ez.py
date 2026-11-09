import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import chirality, geometry, prediction_molecule, read_pdb
from probe_reference_parity import load_encoder, load_features, rotations
from rdkit import Chem
from rgi_benchmark import stereo_matches


def rotate_fragment(molecule, bond):
    left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    fragment = {right}
    pending = [right]
    while pending:
        current = pending.pop()
        for atom in molecule.GetAtomWithIdx(current).GetNeighbors():
            neighbor = atom.GetIdx()
            if {current, neighbor} == {left, right}:
                continue
            if neighbor not in fragment:
                fragment.add(neighbor)
                pending.append(neighbor)
    if left in fragment:
        return None
    xyz = molecule.GetConformer().GetPositions()
    axis = xyz[right] - xyz[left]
    axis /= np.linalg.norm(axis)
    rotation = 2 * np.outer(axis, axis) - np.eye(3)
    changed = xyz.copy()
    indices = sorted(fragment)
    changed[indices] = (xyz[indices] - xyz[left]) @ rotation + xyz[left]
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    model = load_encoder()
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    by_id = {row["id"]: row for row in manifest["records"]}
    output = Path("artifacts/reference_ez")
    output.mkdir(exist_ok=True)
    report = []
    for task in json.loads(args.plan.read_text())["tasks"]:
        if not task.get("trace_layers"):
            continue
        reference = Chem.MolFromSmiles(by_id[task["id"]]["smiles"])
        selected = [
            bond.GetIdx()
            for bond in reference.GetBonds()
            if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
        ]
        if not selected:
            continue
        root = Path(task["output"])
        features = load_features(root)
        atom_type = torch.bmm(
            features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
        token_mask = features["mol_type"][0] == 3
        positions = features["ref_pos"][0, mask].numpy().copy()
        run = json.loads((root / "run.json").read_text())
        atoms = [atom for atom in read_pdb(root / run["prediction"]) if atom["chain"] == "B"]
        molecule = prediction_molecule(
            by_id[task["id"]]["smiles"],
            [{**a, "xyz": xyz} for a, xyz in zip(atoms, positions, strict=True)],
        )
        index_by_name = {atom.GetProp("name"): atom.GetIdx() for atom in molecule.GetAtoms()}
        feature_order = [index_by_name[atom["name"]] for atom in atoms]
        source = molecule.GetConformer().GetPositions()
        differences, variances, bonds = [], [], []
        with np.load(Path("artifacts/reference_parity") / f"{task['id']}.npz") as data:
            original_mean = data["parity_means"][0]
        for bond_index in selected:
            changed = rotate_fragment(molecule, molecule.GetBondWithIdx(bond_index))
            if changed is None:
                report.append({"id": task["id"], "bond": bond_index, "status": "ring_skipped"})
                continue
            altered = Chem.Mol(molecule)
            for index, xyz in enumerate(changed):
                altered.GetConformer().SetAtomPosition(index, xyz)
            distortion = geometry(altered, source)
            assert distortion["bond_rms"] < 1e-6 and distortion["angle_rms"] < 1e-4
            assert chirality(reference, altered) == chirality(reference, molecule)
            assert not stereo_matches(molecule, changed)[1]
            ordered = torch.from_numpy(changed[feature_order]).float()
            center = ordered.mean(dim=0, keepdim=True)
            relative = ordered - center
            values = []
            for rotation in rotations():
                features["ref_pos"][0, mask] = relative @ rotation + center
                values.append(model(features)[0, token_mask].numpy())
            values = np.stack(values)
            mean = values.mean(axis=0)
            difference = mean - original_mean
            variation = values - mean[None]
            gap = float(np.linalg.norm(difference, axis=-1).mean())
            rotation_variation = float(np.linalg.norm(variation, axis=-1).mean())
            row = {
                "id": task["id"],
                "bond": bond_index,
                "status": "complete",
                "bond_rms_change": distortion["bond_rms"],
                "angle_rms_change": distortion["angle_rms"],
                "chirality_preserved": True,
                "mean_ez_gap_norm": gap,
                "mean_rotation_variation_norm": rotation_variation,
            }
            report.append(row)
            differences.append(difference)
            variances.append(np.mean(variation**2, axis=0))
            bonds.append(bond_index)
            print(json.dumps(row), flush=True)
        if differences:
            np.savez_compressed(
                output / f"{task['id']}.npz",
                bond_indices=np.array(bonds),
                ez_difference=np.stack(differences),
                rotation_variance=np.stack(variances),
            )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
