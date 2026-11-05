import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads(Path("artifacts/base_provenance.json").read_text())
    expected = provenance["sha256"]
    base = Path(provenance["local_file"])
    if args.original_checkpoint.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Original verification requires the external checkpoint")
    if args.original_checkpoint.samefile(base):
        raise ValueError("The original checkpoint and local base must be distinct files")
    hashes = {"original_checkpoint": digest(args.original_checkpoint)}
    hashes[base.name] = digest(base)
    checks = {
        "original_checkpoint_unchanged": hashes["original_checkpoint"] == expected,
        "local_base_unchanged": hashes[base.name] == expected,
        "local_base_is_regular_file": base.is_file() and not base.is_symlink(),
    }
    for record_path in sorted(Path("artifacts").glob("*.edit.json")):
        record = json.loads(record_path.read_text())
        checkpoint = Path(record["output"])
        hashes[checkpoint.name] = digest(checkpoint)
        checks[checkpoint.name] = (
            checkpoint.parent == Path(".")
            and checkpoint.is_file()
            and not checkpoint.is_symlink()
            and hashes[checkpoint.name] == record["output_sha256"]
            and record["base_sha256"] == expected
            and record["all_untargeted_tensors_equal"]
        )
    for audit_path in sorted(Path("artifacts").glob("*.pair_audit.json")):
        audit = json.loads(audit_path.read_text())
        checks[f"{audit_path.stem}_paired_conditions_match"] = (
            audit["pairs"] > 0 and audit["pairs"] == audit["passed"]
        )
    assessment = json.loads(Path("artifacts/validation_assessment.json").read_text())
    checks["validation_selection_record_unchanged"] = (
        digest(Path("artifacts/validation_candidate.json")) == assessment["selection_sha256"]
    )
    checks["validation_scores_unchanged"] = (
        digest(Path("artifacts/validation_v1.scores.json")) == assessment["scores_sha256"]
    )
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    test_ids = {row["id"] for row in manifest["records"] if row["split"] == "test"}
    test_tasks = []
    for plan_path in sorted(Path("artifacts").glob("*.json")):
        plan = json.loads(plan_path.read_text())
        if not isinstance(plan, dict):
            continue
        for task in plan.get("tasks", []):
            if task["id"] in test_ids:
                test_tasks.append({"plan": plan_path.name, "id": task["id"]})
    checks["test_split_not_scheduled"] = not test_tasks
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "sha256": hashes,
        "test_tasks": test_tasks,
        "validation_status": assessment["status"],
    }
    Path("artifacts/final_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
