import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate import cip_labels, named_molecule, prediction_molecule, read_pdb


def mapped_labels(reference, predicted):
    reference, expected = cip_labels(reference)
    predicted, observed = cip_labels(predicted, from_3d=True)
    mappings = predicted.GetSubstructMatches(
        reference, useChirality=False, uniquify=False, maxMatches=100000
    )
    if not mappings:
        raise ValueError("No complete graph mapping for center analysis")
    mapping = max(
        mappings,
        key=lambda candidate: (
            sum(observed.get(candidate[index]) == label for index, label in expected.items()),
            sum(index == mapped for index, mapped in enumerate(candidate)),
        ),
    )
    return {index: observed.get(mapping[index]) for index in expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    records = {row["id"]: row for row in manifest["records"]}
    scores = json.loads(args.scores.read_text())
    base = {row["id"]: row for row in scores if row["checkpoint"] == "boltz2_base.ckpt"}
    edited = {row["id"]: row for row in scores if row["checkpoint"] == args.checkpoint}
    if base.keys() != edited.keys():
        raise ValueError("Incomplete paired system coverage")
    centers = []
    failures = []
    for sid in sorted(base):
        if records[sid]["split"] != "development":
            raise ValueError("Center-change diagnosis is restricted to development")
        if base[sid]["status"] != "complete" or edited[sid]["status"] != "complete":
            failures.append(sid)
            continue
        reference, expected = cip_labels(named_molecule(records[sid]["smiles"]))
        observed = []
        for row in (base[sid], edited[sid]):
            root = Path(row["output"])
            run = json.loads((root / "run.json").read_text())
            if run.get("diagnostic") or run.get("reference_reflection"):
                raise ValueError("Diagnostic predictions cannot enter paired center comparisons")
            molecule = prediction_molecule(records[sid]["smiles"], read_pdb(root / run["prediction"]))
            labels = mapped_labels(reference, molecule)
            correct = sum(labels[index] == label for index, label in expected.items())
            if correct != row["chiral_correct"]:
                raise ValueError("Center mapping disagrees with the primary chirality evaluator")
            observed.append(labels)
        for index, label in expected.items():
            atom = reference.GetAtomWithIdx(index)
            centers.append({
                "id": sid,
                "atom_name": atom.GetProp("name"),
                "element": atom.GetSymbol(),
                "heavy_atom_degree": atom.GetDegree(),
                "in_ring": atom.IsInRing(),
                "neighbor_elements": sorted(neighbor.GetSymbol() for neighbor in atom.GetNeighbors()),
                "expected": label,
                "base": observed[0][index],
                "edited": observed[1][index],
                "base_correct": observed[0][index] == label,
                "edited_correct": observed[1][index] == label,
            })
    groups = defaultdict(list)
    for center in centers:
        key = (center["element"], center["heavy_atom_degree"], center["in_ring"], tuple(center["neighbor_elements"]))
        groups[key].append(center)
    summaries = []
    for key, rows in sorted(groups.items()):
        summaries.append({
            "element": key[0], "heavy_atom_degree": key[1], "in_ring": key[2],
            "neighbor_elements": list(key[3]), "centers": len(rows),
            "base_correct": sum(row["base_correct"] for row in rows),
            "edited_correct": sum(row["edited_correct"] for row in rows),
            "improved": sum(not row["base_correct"] and row["edited_correct"] for row in rows),
            "worsened": sum(row["base_correct"] and not row["edited_correct"] for row in rows),
        })
    report = {
        "scope": "Descriptive development center changes; not an independent validation or a new acceptance metric.",
        "mapping": "Maximize correct R/S centers over graph isomorphisms, as in the primary evaluator; ties prefer unchanged atom indices. Symmetric-center attribution can remain ambiguous.",
        "scores": args.scores.as_posix(), "checkpoint": args.checkpoint,
        "systems": len(base), "failed_systems": failures,
        "groups": summaries, "centers": centers,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("systems", "failed_systems", "groups")}))


if __name__ == "__main__":
    main()
