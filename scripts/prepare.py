import argparse
import csv
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from Bio.Align import PairwiseAligner
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    out = Path("data/ct209")
    if (out / "manifest.json").exists():
        raise FileExistsError(out / "manifest.json")
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "msa").mkdir(exist_ok=True)
    (out / "sdf").mkdir(exist_ok=True)
    rows = list(csv.DictReader((args.dataset / "index_subct209.csv").open()))
    assert len(rows) == 209
    records = []
    for index, row in enumerate(rows):
        sid = row["sys_id"]
        source = args.dataset / f"smict209/b_{index // 10}/inputs/{sid}.yaml"
        if not source.exists():
            matches = list((args.dataset / "smict209").glob(f"b_*/inputs/{sid}.yaml"))
            assert len(matches) == 1, sid
            source = matches[0]
        doc = yaml.safe_load(source.read_text())
        prot = next(item["protein"] for item in doc["sequences"] if "protein" in item)
        lig = next(item["ligand"] for item in doc["sequences"] if "ligand" in item)
        assert lig["smiles"] == row["smiles"]
        assert lig.get("conformer_restraints") is False
        assert "constraints" not in doc and "templates" not in doc
        lig.pop("conformer_restraints")
        msa = (source.parent / prot["msa"]).resolve()
        sdf = args.dataset / "hf/plinder_cistrans_209/sdf" / f"{sid}.sdf"
        for src, dest, hash_key in (
            (msa, out / "msa" / f"{sid}.a3m", "a3m_sha256"),
            (sdf, out / "sdf" / f"{sid}.sdf", "sdf_sha256"),
        ):
            assert digest(src) == row[hash_key], (sid, hash_key)
            shutil.copyfile(src, dest)
        prot["msa"] = f"../msa/{sid}.a3m"
        dest = out / "inputs" / f"{sid}.yaml"
        dest.write_text(yaml.safe_dump(doc, sort_keys=False))
        mol = Chem.MolFromSmiles(row["smiles"])
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            scaffold = Chem.MolToSmiles(mol, isomericSmiles=False)
        records.append(
            {
                "id": sid,
                "smiles": row["smiles"],
                "sequence": prot["sequence"],
                "scaffold": scaffold,
                "input": dest.as_posix(),
                "input_sha256": digest(dest),
                "msa_sha256": row["a3m_sha256"],
                "sdf_sha256": row["sdf_sha256"],
                "source_input_sha256": digest(source),
                "ligand_atoms": mol.GetNumHeavyAtoms(),
                "protein_length": len(prot["sequence"]),
            }
        )
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    aligner = PairwiseAligner(
        mode="global", match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.05
    )
    for i, left in enumerate(records):
        for j in range(i):
            right = records[j]
            same = left["scaffold"] == right["scaffold"]
            if not same:
                a, b = left["sequence"], right["sequence"]
                if min(len(a), len(b)) / max(len(a), len(b)) >= 0.8:
                    aln = aligner.align(a, b)[0]
                    same = aln.counts().identities / max(len(a), len(b)) >= 0.4
            if same:
                parents[root(i)] = root(j)
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(root(i), []).append(record)
    ordered = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(
            ("wsp-boltz2-v1:" + min(x["id"] for x in group)).encode()
        ).hexdigest(),
    )
    counts = {"development": 0, "validation": 0, "test": 0}
    for group in ordered:
        split = (
            "development"
            if counts["development"] < 42
            else ("validation" if counts["validation"] < 32 else "test")
        )
        group_id = min(x["id"] for x in group)
        for record in group:
            record.update(split=split, group=group_id)
        counts[split] += len(group)
    pilot = sorted(
        (r for r in records if r["split"] == "development"),
        key=lambda r: (r["protein_length"], r["id"]),
    )[:8]
    manifest = {
        "dataset": "bench-rgi ct209",
        "index_sha256": digest(args.dataset / "index_subct209.csv"),
        "split_rule": "Connected components: equal achiral Murcko scaffold or global sequence identity >=0.4 of longer sequence with length ratio >=0.8; deterministic group order; 42 development, 32 validation minimum; remainder test.",
        "counts": counts,
        "pilot_ids": [r["id"] for r in pilot],
        "records": records,
        "versions": {k: version(k) for k in ("boltz", "torch", "rdkit", "biopython")},
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps({"counts": counts, "pilot": [(r["id"], r["protein_length"]) for r in pilot]}),
        flush=True,
    )
    before = digest(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt["state_dict"]
    stats = {}
    for key, value in state.items():
        if any(
            name in key
            for name in (
                "embed_atom",
                "atom_feat_to_atom_pos",
                "r_to_q",
                "z_to_p_trans",
                "s_to_c_trans",
            )
        ):
            v = value.float()
            stats[key] = {
                "shape": list(v.shape),
                "norm": v.norm().item(),
                "mean": v.mean().item(),
                "std": v.std().item(),
            }
            if "embed_atom_features.weight" in key:
                stats[key]["column_norms"] = v.norm(dim=0).tolist()
            elif v.numel() < 512:
                stats[key]["values"] = v.tolist()
    write_json(
        Path("artifacts/weight_inspection.json"),
        {
            "base_sha256": before,
            "parameters": sum(x.numel() for x in state.values()),
            "hyper_parameters": {k: str(v) for k, v in ckpt["hyper_parameters"].items()},
            "stats": stats,
        },
    )
    shutil.copyfile(args.checkpoint, "boltz2_base.ckpt")
    assert digest(Path("boltz2_base.ckpt")) == before == digest(args.checkpoint)
    write_json(
        Path("artifacts/base_provenance.json"),
        {"sha256": before, "local_file": "boltz2_base.ckpt", "source_unchanged": True},
    )
    print("Preparation and base checkpoint verification complete", flush=True)


if __name__ == "__main__":
    main()
