import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import named_molecule, read_pdb
from probe_reference_parity import load_encoder, load_features, rotations
from program_neighbor_bias import program_neighbor_bias
from program_neighborhood_mask import program_neighborhood_mask


class AttentionRecorder:
    def __init__(self, module, indices, width):
        self.windows = torch.tensor([index // width for index in indices])
        self.queries = torch.tensor([index % width for index in indices])
        self.weights = None
        self.handle = module.register_forward_pre_hook(self.capture, with_kwargs=True)

    def capture(self, module, inputs, named):
        count = named["s"].shape[0]
        heads, dimension = module.num_heads, module.head_dim
        query = module.proj_q(named["s"]).view(count, -1, heads, dimension)
        keys = module.proj_k(named["k_in"]).view(count, -1, heads, dimension)
        query = query[self.windows, self.queries]
        keys = keys[self.windows]
        bias = module.proj_z(named["z"])[self.windows, :, self.queries]
        mask = named["mask"][self.windows]
        logits = torch.einsum("chd,ckhd->chk", query.float(), keys.float()) / dimension**0.5
        logits += bias.float() + (1 - mask[:, None].float()) * -module.inf
        self.weights = logits.softmax(-1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--centers", type=Path, default=Path("artifacts/persistent_centers.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/neighbor_attention.json"))
    parser.add_argument("--program-neighbor-bias", action="store_true")
    parser.add_argument("--strict-mask", action="store_true")
    parser.add_argument("--layer", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = load_encoder()
    if args.program_neighbor_bias and args.strict_mask:
        parser.error("Choose only one programmed attention diagnostic")
    if args.strict_mask:
        model.load_state_dict(program_neighborhood_mask(model.state_dict()), strict=False)
    elif args.program_neighbor_bias:
        model.load_state_dict(program_neighbor_bias(model.state_dict()), strict=False)
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {row["id"]: row for row in manifest["records"]}
    tasks = {
        task["id"]: task for task in json.loads(args.plan.read_text())["tasks"]
        if task["checkpoint"] == "boltz2_base.ckpt" and task.get("trace_layers")
    }
    report = []
    for system in json.loads(args.centers.read_text())["systems"]:
        sid = system["id"]
        if records[sid]["split"] != "development":
            raise ValueError("Attention diagnosis is restricted to development")
        centers = [c for c in system["centers"] if c["never_correct_in_observed_predictions"]]
        if not centers:
            continue
        root = Path(tasks[sid]["output"])
        run = json.loads((root / "run.json").read_text())
        atoms = [a for a in read_pdb(root / run["prediction"]) if a["chain"] == "B"]
        features = load_features(root)
        atom_type = torch.bmm(
            features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        ligand_mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
        ligand_indices = torch.nonzero(ligand_mask).flatten().tolist()
        global_by_name = {
            atom["name"]: index for atom, index in zip(atoms, ligand_indices, strict=True)
        }
        molecule = named_molecule(records[sid]["smiles"])
        global_centers = [global_by_name[center["atom_name"]] for center in centers]
        neighbors = []
        for center in centers:
            atom = molecule.GetAtomWithIdx(center["atom_index"])
            if atom.GetDegree() != 3:
                raise ValueError("This diagnostic requires three heavy-atom neighbors")
            neighbors.append([
                global_by_name[neighbor.GetProp("name")]
                for neighbor in sorted(atom.GetNeighbors(), key=lambda neighbor: neighbor.GetIdx())
            ])
        _, _, _, to_keys = model.atom_encoder(features)
        width = model.atom_encoder.atoms_per_window_queries
        key_width = model.atom_encoder.atoms_per_window_keys
        indices = torch.arange(1, features["ref_pos"].shape[1] + 1).reshape(1, -1, 1).float()
        key_indices = to_keys(indices).reshape(-1, key_width).long().numpy() - 1
        neighbor_columns = []
        self_columns = []
        for global_index, adjacent in zip(global_centers, neighbors, strict=True):
            keys = key_indices[global_index // width]
            neighbor_columns.append([int(np.flatnonzero(keys == neighbor)[0]) for neighbor in adjacent])
            self_columns.append(int(np.flatnonzero(keys == global_index)[0]))
        module = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[args.layer].pair_bias_attn
        recorder = AttentionRecorder(module, global_centers, width)
        positions = features["ref_pos"].clone()
        center = positions[0, ligand_mask].mean(0, keepdim=True)
        relative = positions[0, ligand_mask] - center
        model(features)
        ordinary = recorder.weights.copy()
        selected = []
        moment_determinants = []
        head_triples = list(itertools.combinations(range(module.num_heads), 3))
        for parity in (1.0, -1.0):
            weight_group = []
            volume_group = []
            for rotation in rotations():
                features["ref_pos"][0, ligand_mask] = parity * relative @ rotation + center
                model(features)
                weights = np.stack([
                    recorder.weights[index][:, columns]
                    for index, columns in enumerate(neighbor_columns)
                ])
                weight_group.append(weights)
                volumes = []
                for index, (global_index, adjacent) in enumerate(zip(global_centers, neighbors, strict=True)):
                    vectors = (features["ref_pos"][0, adjacent] - features["ref_pos"][0, global_index]).numpy()
                    moments = weights[index] @ vectors
                    volumes.append([float(np.linalg.det(moments[list(triple)])) for triple in head_triples])
                volume_group.append(volumes)
            selected.append(weight_group)
            moment_determinants.append(volume_group)
        recorder.handle.remove()
        weights = np.asarray(selected)
        volumes = np.asarray(moment_determinants)
        rows = []
        for index, atom in enumerate(centers):
            normal = ordinary[index]
            columns = neighbor_columns[index]
            neighborhood = normal[:, columns]
            singular = np.linalg.svd(neighborhood, compute_uv=False)
            rows.append({
                "atom_name": atom["atom_name"],
                "expected_cip": atom["expected_cip"],
                "attention_mass_on_self_per_head": normal[:, self_columns[index]].tolist(),
                "attention_mass_on_neighbors_per_head": neighborhood.sum(-1).tolist(),
                "neighbor_weight_singular_values": singular.tolist(),
                "neighbor_weights": neighborhood.tolist(),
                "neighbor_weight_max_rotation_range": float(np.ptp(weights[:, :, index], axis=1).max()),
                "head_triples": [
                    {
                        "heads": list(triple),
                        "minimum_absolute_moment_determinant": float(np.abs(volumes[:, :, index, item]).min()),
                        "proper_rotation_positive_count": int((volumes[0, :, index, item] > 0).sum()),
                        "reflected_rotation_positive_count": int((volumes[1, :, index, item] > 0).sum()),
                        "paired_reflection_opposite_sign_count": int((volumes[0, :, index, item] * volumes[1, :, index, item] < 0).sum()),
                        "rotation_count": 24,
                    }
                    for item, triple in enumerate(head_triples)
                ],
            })
        report.append({"id": sid, "centers": rows})
        print(json.dumps(report[-1]), flush=True)
    args.output.write_text(json.dumps({
        "scope": "Input-encoder attention at persistent development chirality errors.",
        "layer": args.layer,
        "programmed_neighbor_bias": args.program_neighbor_bias,
        "strict_neighborhood_mask": args.strict_mask,
        "limitation": "Neighbor-restricted geometric moments are descriptive. They are not the learned value vectors, do not establish a downstream readout, and do not constitute a repair.",
        "systems": report,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
