import argparse
import json
from pathlib import Path

import torch
from boltz.data import const
from boltz.data.tokenize.boltz2 import tokenize_structure
from boltz.data.types import StructureV2
from checkpoint_io import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    state = load_checkpoint("boltz2_base.ckpt", map_location="cpu", mmap=True)[
        "state_dict"
    ]
    embedding = state["token_bonds_type.weight"].double()
    feature_ids = {name: index + 1 for name, index in const.bond_type_ids.items()}
    single = embedding[feature_ids["SINGLE"]]
    weights = {
        name: {
            "index": index,
            "norm": float(embedding[index].norm()),
            "difference_from_single_norm": float((embedding[index] - single).norm()),
        }
        for name, index in feature_ids.items()
    }
    systems = []
    for task in json.loads(args.plan.read_text())["tasks"]:
        if task["checkpoint"] != "boltz2_base.ckpt":
            continue
        root = Path(task["output"])
        processed = root / "prediction/boltz_results_target/processed"
        structure = StructureV2.load(processed / "structures/target.npz")
        tokens, bonds = tokenize_structure(structure, None)
        ligand = {int(token["token_idx"]): token["mol_type"] == 3 for token in tokens}
        pairs = {
            tuple(sorted((int(bond["token_1"]), int(bond["token_2"])))): int(bond["type"])
            for bond in bonds
            if bond["token_1"] != bond["token_2"]
        }
        counts = {name: {"ligand_pairs": 0, "other_pairs": 0} for name in feature_ids}
        names = {value: key for key, value in feature_ids.items()}
        for (left, right), value in pairs.items():
            scope = "ligand_pairs" if ligand[left] and ligand[right] else "other_pairs"
            counts[names[value]][scope] += 1
        systems.append({"id": task["id"], "counts": counts})
    report = {
        "feature_index_rule": "The tokenizer adds one to chemical bond IDs; zero is no bond.",
        "embedding_shape": list(embedding.shape),
        "weights": weights,
        "systems": systems,
    }
    Path("artifacts/bond_type_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"weights": weights, "systems": len(systems)}))


if __name__ == "__main__":
    main()
