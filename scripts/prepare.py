import argparse
import csv
import hashlib
import json
import lzma
import shutil
from importlib.metadata import version
from pathlib import Path

import yaml
from Bio.Align import PairwiseAligner
from checkpoint_io import load_checkpoint
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


DATASET_ID = "wasarou/plinder-cistrans-209"
DATASET_REVISION = "c99d4646cbf2c8359ea00d6bc647a62be085c5bd"
INDEX_SHA256 = "50fe82089aaba7c9126624b50338a3fa5eead2a04c762780834651391f57632a"


def msa_query(text):
    sequence = []
    started = False
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started:
            sequence.append(line.strip())
        else:
            raise ValueError("Invalid A3M")
    query = "".join(c for c in "".join(sequence) if c.isupper())
    if not query:
        raise ValueError("Empty A3M query")
    return query


def public_inputs(dataset):
    """Yield verified rows, MSAs, crystal SDFs, and Boltz inputs from the public dataset."""
    if digest(dataset / "index.csv") != INDEX_SHA256:
        raise ValueError(f"Use {DATASET_ID} at revision {DATASET_REVISION}")
    rows = list(csv.DictReader((dataset / "index.csv").open()))
    if len(rows) != 209:
        raise ValueError("Expected 209 systems")
    for row in rows:
        sid = row["sys_id"]
        msa = lzma.decompress((dataset / "msa" / f"{sid}.a3m.xz").read_bytes())
        sdf = dataset / "sdf" / f"{sid}.sdf"
        if hashlib.sha256(msa).hexdigest() != row["a3m_sha256"] or digest(sdf) != row["sdf_sha256"]:
            raise ValueError(f"Dataset file hash differs: {sid}")
        doc = {
            "sequences": [
                {"protein": {"id": "A", "msa": f"../msa/{sid}.a3m",
                             "sequence": msa_query(msa.decode())}},
                {"ligand": {"id": "B", "smiles": row["smiles"]}},
            ],
            "version": 1,
        }
        yield row, msa, sdf, doc


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
    records = []
    for row, msa, sdf, doc in public_inputs(args.dataset):
        sid = row["sys_id"]
        (out / "msa" / f"{sid}.a3m").write_bytes(msa)
        shutil.copyfile(sdf, out / "sdf" / f"{sid}.sdf")
        prot = doc["sequences"][0]["protein"]
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
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "index_sha256": INDEX_SHA256,
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
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu", mmap=True)
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
