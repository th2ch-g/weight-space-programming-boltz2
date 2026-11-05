"""Freeze v11 for follow-up evaluation without claiming an untouched test set."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-protein-threshold", type=int, default=600)
    args = parser.parse_args()
    if args.output.is_absolute() or args.output.exists():
        raise ValueError("Use a new relative study directory")
    baseline = "boltz2_base.ckpt"
    candidate = "boltz2_wsp_absolute_vectors_only_v11.ckpt"
    edit = json.loads(Path("artifacts/boltz2_wsp_absolute_vectors_only_v11.edit.json").read_text())
    expected = {baseline: edit["base_sha256"], candidate: edit["output_sha256"]}
    hashes = {}
    for name, checksum in expected.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected a regular local checkpoint")
        hashes[name] = digest(path)
        if hashes[name] != checksum:
            raise ValueError(f"Checkpoint changed: {name}")
    manifest_path = Path("data/ct209/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    records = {r["id"]: r for r in manifest["records"]}
    if Counter(r["split"] for r in records.values()) != {
        "development": 55, "validation": 50, "test": 104,
    }:
        raise ValueError("Unexpected dataset split")
    for row in records.values():
        for path, checksum in (
            (Path(row["input"]), row["input_sha256"]),
            (Path("data/ct209/msa") / f"{row['id']}.a3m", row["msa_sha256"]),
            (Path("data/ct209/sdf") / f"{row['id']}.sdf", row["sdf_sha256"]),
        ):
            if digest(path) != checksum:
                raise ValueError(f"Dataset input changed: {path}")
    source_scores = Path("artifacts/all_candidates_development/plan.scores.json")
    old_rows = json.loads(source_scores.read_text())
    development = [r for r in old_rows if r["checkpoint"] in expected]
    by_key = {(r["id"], r["checkpoint"]): r for r in development}
    if len(by_key) != len(development) or len(development) != 110:
        raise ValueError("Unexpected development prediction coverage")
    reused_runs = {}
    for row in development:
        if records[row["id"]]["split"] != "development" or row["status"] != "complete":
            raise ValueError("Only completed development predictions may be reused")
        run_path = Path(row["output"]) / "run.json"
        run = json.loads(run_path.read_text())
        if run["checkpoint_sha256"] != hashes[row["checkpoint"]] or run["seed"] != 1:
            raise ValueError("Reused prediction has the wrong checkpoint or seed")
        if run.get("diagnostic") or run.get("reference_reflection"):
            raise ValueError("Diagnostic predictions cannot enter this comparison")
        prediction = run_path.parent / run["prediction"]
        if digest(prediction) != run["prediction_sha256"]:
            raise ValueError("Reused prediction changed")
        reused_runs[row["output"]] = digest(run_path)
    for sid in {r["id"] for r in development}:
        runs = [json.loads((Path(by_key[sid, c]["output"]) / "run.json").read_text())
                for c in expected]
        for key in ("seed", "settings", "versions", "runner_sha256", "source_sha256", "gpu", "job_id"):
            if runs[0].get(key) != runs[1].get(key):
                raise ValueError(f"Reused pair differs in {key}")
    tasks = []
    analysis_plans = {}
    for split, seeds in (("validation", [1]), ("test", [1, 2, 3])):
        selected = sorted(
            (r for r in records.values() if r["split"] == split),
            key=lambda r: (-r["protein_length"], r["id"]),
        )
        for seed in seeds:
            selected_tasks = []
            for row in selected:
                for checkpoint, arm in ((baseline, "base"), (candidate, "edited")):
                    task = {
                        "id": row["id"], "checkpoint": checkpoint, "seed": seed,
                        "output": (args.output / "runs" / f"seed_{seed}" / arm / row["id"]).as_posix(),
                        "diagnostic": False,
                    }
                    tasks.append(task)
                    selected_tasks.append(task)
            name = f"{split}_seed_{seed}"
            analysis_plans[name] = {"stage": split, "tasks": selected_tasks}
    analysis_plans["development_seed_1"] = {
        "stage": "development",
        "tasks": [{k: r[k] for k in ("id", "checkpoint", "output")} | {"seed": 1}
                  for r in development],
    }
    assert len(tasks) == 724
    assert len({t["output"] for t in tasks}) == len(tasks)
    source_names = [
        "scripts/predict.py", "scripts/worker.py", "scripts/evaluate.py", "scripts/chemistry.py",
        "scripts/audit_pairs.py", "scripts/summarize.py", "scripts/slurm.sh",
        "scripts/prepare_v11_evaluation.py", "scripts/report_v11_evaluation.py",
        "scripts/report_frozen_evaluation.py", "tests/test_v11_evaluation.py",
        "pyproject.toml", "uv.lock",
        "protocol.json", "goal_protocol.json", "data/ct209/manifest.json",
        "edits/absolute_vectors_only_v11.json",
    ]
    source_hashes = {name: digest(name) for name in source_names}
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "Characterize the frozen v11 checkpoint requested after the matched thirty-candidate development comparison; do not search for new weights.",
        "authorization": "The user explicitly requested the same validation and test evaluation for v11.",
        "baseline": baseline, "candidate": candidate, "checkpoint_sha256": hashes,
        "manifest_sha256": digest(manifest_path), "source_sha256": source_hashes,
        "all_209_input_msa_sdf_hashes_verified": True,
        "primary_population": "The existing 104-system test split, seeds 1, 2, and 3. This split has already been evaluated with v2.",
        "secondary_populations": "Validation 50 systems at seed 1; all 209 systems at seed 1, including reused development predictions.",
        "reused_development_scores": source_scores.as_posix(),
        "reused_development_scores_sha256": digest(source_scores),
        "reused_run_sha256": reused_runs,
        "inference": json.loads(Path("protocol.json").read_text())["inference"],
        "metrics": ["chirality_all", "full_ez_all", "full_stereo_all", "ideal_bond_rms", "ideal_angle_rms"],
        "other_metrics": "Report legacy E/Z, per-center errors, means, regressions, clashes, and paired protein displacement.",
        "uncertainty": "10000 paired bootstrap draws of sequence/scaffold groups; seed 20261103. Pool repeated test seeds by averaging within each system before resampling groups; also report each seed separately.",
        "geometry_reference": "Independent ETKDGv3 plus UFF conformer, fixed seed 20261103; not experimental pose error.",
        "test_policy": "The v2 test results were already known before this follow-up. This is not a newly untouched test. Freeze v11 and all endpoints before its validation/test predictions, and do not use those results for further checkpoint selection or editing. Three seeds do not establish robustness to arbitrary seeds.",
        "prior_test_report": "artifacts/frozen_v2_evaluation/report.json",
        "prior_test_report_sha256": digest("artifacts/frozen_v2_evaluation/report.json"),
        "original_acceptance": json.loads(Path("goal_protocol.json").read_text())["primary_acceptance"],
        "acceptance_policy": "Preserve and report the original zero-error criteria separately. This characterization is authorized even though the candidate already fails that repair gate on development.",
        "failures": "Retain all attempts. Retry only identified execution failures with the same frozen settings; do not omit failed or unverifiable systems.",
        "resources": {"max_concurrent_gpu_nodes": 4, "standard_workers": 3, "large_workers": 1,
                      "large_protein_threshold": args.large_protein_threshold},
        "planned_new_predictions": len(tasks),
    }
    args.output.mkdir(parents=True)
    for name, plan in analysis_plans.items():
        plan["manifest_sha256"] = protocol["manifest_sha256"]
        write_json(args.output / "plans" / f"{name}.json", plan)
    for name, large in (("standard", False), ("large", True)):
        selected = [t for t in tasks if (records[t["id"]]["protein_length"] > args.large_protein_threshold) == large]
        write_json(args.output / "plans" / f"predict_{name}.json", {
            "stage": "frozen_evaluation", "manifest_sha256": protocol["manifest_sha256"],
            "tasks": selected,
        })
    write_json(args.output / "plan_manifest.json", {
        path.as_posix(): digest(path) for path in sorted((args.output / "plans").glob("*.json"))
    })
    write_json(args.output / "analysis_manifest.json", {
        name: digest(name) for name in (
            "scripts/report_frozen_evaluation.py", "scripts/report_v11_evaluation.py",
            "tests/test_v11_evaluation.py", "../impl_rgi/bench-rgi/common/chiral.py",
        )
    })
    with tarfile.open(args.output / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in source_names:
            archive.add(name, arcname=name)
    protocol["source_snapshot_sha256"] = digest(args.output / "source_snapshot.tar.gz")
    write_json(args.output / "protocol.json", protocol)
    print(json.dumps({"study": args.output.as_posix(), "new_predictions": len(tasks),
                      "reused_predictions": len(development), "checkpoint_hashes_verified": True}))


if __name__ == "__main__":
    main()
