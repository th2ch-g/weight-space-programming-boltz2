import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem


def arrays_equal(left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.names:
        return all(arrays_equal(left[key], right[key]) for key in left.dtype.names)
    if left.dtype.kind == "O":
        return all(objects_equal(a, b) for a, b in zip(left.flat, right.flat, strict=True))
    return np.array_equal(left, right, equal_nan=left.dtype.kind in "fc")


def objects_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return arrays_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            objects_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            objects_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def npz_equal(left, right):
    with np.load(left, allow_pickle=True) as a, np.load(right, allow_pickle=True) as b:
        return set(a.files) == set(b.files) and all(arrays_equal(a[key], b[key]) for key in a.files)


def mols_equal(left, right):
    with left.open("rb") as handle:
        a = pickle.load(handle)
    with right.open("rb") as handle:
        b = pickle.load(handle)
    if a.keys() != b.keys():
        return False
    for key in a:
        x, y = a[key], b[key]
        if Chem.MolToSmiles(x) != Chem.MolToSmiles(y):
            return False
        if [atom.GetProp("name") for atom in x.GetAtoms()] != [
            atom.GetProp("name") for atom in y.GetAtoms()
        ]:
            return False
        if x.GetNumConformers() != y.GetNumConformers():
            return False
        for cx, cy in zip(x.GetConformers(), y.GetConformers(), strict=True):
            if not np.array_equal(cx.GetPositions(), cy.GetPositions()):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.plan.read_text())["tasks"]
    base = {t["id"]: t for t in tasks if t["checkpoint"] == "boltz2_base.ckpt"}
    audit = []
    for task in tasks:
        if task["checkpoint"] == "boltz2_base.ckpt":
            continue
        original = base[task["id"]]
        roots = [Path(t["output"]) for t in (original, task)]
        records = [json.loads((root / "run.json").read_text()) for root in roots]
        if any(record["status"] != "complete" for record in records):
            audit.append({"id": task["id"], "status": "incomplete"})
            continue
        checks = {
            key: records[0].get(key) == records[1].get(key)
            for key in (
                "id",
                "seed",
                "input_sha256",
                "msa_sha256",
                "settings",
                "versions",
                "runner_sha256",
                "checkpoint_loader_sha256",
                "source_sha256",
                "gpu",
                "job_id",
                "local_molecule_sha256",
                "reference_reflection",
            )
        }
        processed = [root / "prediction/boltz_results_target/processed" for root in roots]
        for relative in ("structures/target.npz", "msa/target_0.npz", "constraints/target.npz"):
            a, b = [root / relative for root in processed]
            checks[relative] = (
                npz_equal(a, b) if a.exists() and b.exists() else not a.exists() and not b.exists()
            )
        a, b = [root / "mols/target.pkl" for root in processed]
        checks["reference_conformers"] = (
            mols_equal(a, b) if a.exists() and b.exists() else not a.exists() and not b.exists()
        )
        audit.append(
            {
                "id": task["id"],
                "checkpoint": task["checkpoint"],
                "status": "pass" if all(checks.values()) else "mismatch",
                "checks": checks,
            }
        )
    report = {
        "pairs": len(audit),
        "passed": sum(row["status"] == "pass" for row in audit),
        "rows": audit,
    }
    args.plan.with_suffix(".pair_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pairs": report["pairs"],
                "passed": report["passed"],
                "failures": [row for row in audit if row["status"] != "pass"],
            }
        )
    )
    if report["pairs"] != report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
