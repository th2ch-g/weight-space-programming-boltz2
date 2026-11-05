"""Compare all retained candidates on the same development systems and baseline."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from summarize import compare

BASE = "boltz2_base.ckpt"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def version_number(name):
    return int(re.search(r"_v(\d+)\.ckpt$", name)[1])


def nondominated(vectors):
    """Return keys not strictly dominated when every coordinate is maximized."""
    return [key for key, vector in vectors.items() if not any(
        other != key and all(a >= b for a, b in zip(values, vector, strict=True))
        and any(a > b for a, b in zip(values, vector, strict=True))
        for other, values in vectors.items()
    )]


def prepare(root):
    if root.is_absolute() or root.exists():
        raise ValueError("Use a new relative study directory")
    candidates = sorted((p.name for p in Path.cwd().glob("boltz2_wsp_*.ckpt")),
                        key=version_number)
    if [version_number(name) for name in candidates] != list(range(1, 31)):
        raise ValueError("Expected exactly the thirty retained candidates")
    expected_base = json.loads(Path("artifacts/base_provenance.json").read_text())["sha256"]
    hashes = {BASE: digest(BASE)}
    if hashes[BASE] != expected_base or Path(BASE).is_symlink():
        raise ValueError("The original local baseline changed")
    for name in candidates:
        record = json.loads(Path("artifacts", name.removesuffix(".ckpt") + ".edit.json").read_text())
        if Path(name).is_symlink() or record["base_sha256"] != expected_base:
            raise ValueError("Candidate provenance differs")
        if not record["all_untargeted_tensors_equal"]:
            raise ValueError("Candidate changed unapproved tensors")
        hashes[name] = digest(name)
        if hashes[name] != record["output_sha256"]:
            raise ValueError(f"Candidate changed: {name}")
    manifest_path = Path("data/ct209/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    records = sorted((r for r in manifest["records"] if r["split"] == "development"),
                     key=lambda r: (-r["protein_length"], r["id"]))
    if len(records) != 55:
        raise ValueError("Unexpected development population")
    tasks = []
    for row in records:
        for name in [BASE, *candidates]:
            arm = "base" if name == BASE else f"v{version_number(name)}"
            tasks.append({
                "id": row["id"], "checkpoint": name, "seed": 1, "diagnostic": False,
                "output": (root / "runs" / arm / row["id"]).as_posix(),
            })
    assert len(tasks) == 1705 and len({t["output"] for t in tasks}) == 1705
    source_names = [
        "scripts/predict.py", "scripts/worker.py", "scripts/evaluate.py", "scripts/chemistry.py",
        "scripts/audit_pairs.py", "scripts/summarize.py", "scripts/slurm.sh",
        "scripts/compare_all_candidates.py", "tests/test_candidate_comparison.py",
        "pyproject.toml", "uv.lock", "data/ct209/manifest.json",
    ]
    root.mkdir(parents=True)
    plan = {"stage": "development", "manifest_sha256": digest(manifest_path), "tasks": tasks}
    write_json(root / "plan.json", plan)
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "Evaluate every retained candidate under one matched development comparison before making any best-candidate claim.",
        "population": "The same pre-existing 55 development systems; seed 1 only.",
        "candidates": candidates, "baseline": BASE, "checkpoint_sha256": hashes,
        "plan_sha256": digest(root / "plan.json"),
        "source_sha256": {name: digest(name) for name in source_names},
        "inference": json.loads(Path("protocol.json").read_text())["inference"],
        "comparison": "For each system, evaluate the baseline and all thirty checkpoints in the same GPU allocation with identical seed, inputs, software, and guidance disabled. Do not combine historical baselines from different runs.",
        "selection": "Report every endpoint and the non-dominated candidates, including the baseline. Do not invent a scalar score after seeing results. Report separately candidates with higher R/S and joint agreement, nondecreasing full E/Z, and lower bond and angle medians. No failed or incomplete candidate qualifies.",
        "uncertainty": "10000 paired sequence/scaffold group bootstrap draws, seed 20261103. Intervals are descriptive; thirty-candidate selection is not independent confirmation.",
        "test_policy": "Only development systems occur here. The parallel frozen-v2 test experiment does not select or alter these candidates. A development winner is not a validated replacement.",
        "repair_acceptance": "The original 209-system zero-error requirement remains unchanged. This development comparison cannot satisfy it.",
        "resources": {"max_concurrent_gpu_nodes": 4, "workers": 4},
        "failure_policy": "Keep all attempts. Retry identified execution failures only; numerical or chemical failures remain outcomes. Incomplete candidates cannot qualify for selection.",
    }
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in source_names:
            archive.add(name, arcname=name)
    protocol["source_snapshot_sha256"] = digest(root / "source_snapshot.tar.gz")
    write_json(root / "protocol.json", protocol)
    print(json.dumps({"study": str(root), "candidates": len(candidates), "systems": len(records),
                      "predictions": len(tasks), "checkpoint_hashes_verified": True}))


def report(root):
    protocol = json.loads((root / "protocol.json").read_text())
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != protocol["plan_sha256"]:
        raise ValueError("The frozen plan changed")
    for name, checksum in {**protocol["source_sha256"], **protocol["checkpoint_sha256"]}.items():
        if digest(name) != checksum:
            raise ValueError(f"Frozen artifact changed: {name}")
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    expected_ids = {r["id"] for r in manifest["records"] if r["split"] == "development"}
    runs = {}
    for task in plan["tasks"]:
        run_path = Path(task["output"]) / "run.json"
        if not run_path.exists():
            raise ValueError(f"Prediction missing: {task['id']} {task['checkpoint']}")
        run = json.loads(run_path.read_text())
        if run["status"] != "complete":
            raise ValueError(f"Prediction failed: {task['id']} {task['checkpoint']}")
        if run["checkpoint_sha256"] != protocol["checkpoint_sha256"][task["checkpoint"]]:
            raise ValueError("Prediction used a different checkpoint")
        if run["runner_sha256"] != protocol["source_sha256"]["scripts/predict.py"]:
            raise ValueError("Prediction used a different runner")
        if run["seed"] != 1 or run.get("diagnostic") or run.get("reference_reflection"):
            raise ValueError("Unexpected inference intervention or seed")
        if any(run["settings"][key] for key in (
            "physical_guidance_update", "contact_guidance_update", "fk_steering",
        )):
            raise ValueError("Guidance was enabled")
        if digest(run_path.parent / run["prediction"]) != run["prediction_sha256"]:
            raise ValueError("Prediction changed")
        runs[task["id"], task["checkpoint"]] = run
    for sid in expected_ids:
        baseline = runs[sid, BASE]
        for candidate in protocol["candidates"]:
            edited = runs[sid, candidate]
            for key in ("seed", "settings", "versions", "runner_sha256", "source_sha256", "gpu", "job_id"):
                if baseline[key] != edited[key]:
                    raise ValueError(f"Unmatched pair: {key}")
    for script in ("evaluate.py", "audit_pairs.py"):
        subprocess.run([sys.executable, str(Path("scripts") / script), str(plan_path)], check=True)
    scores = json.loads(plan_path.with_suffix(".scores.json").read_text())
    baseline = [r for r in scores if r["checkpoint"] == BASE]
    if len(baseline) != 55 or any(r["status"] != "complete" for r in baseline):
        raise ValueError("The shared baseline evaluation is incomplete")
    comparisons = []
    failed = {}
    for candidate in protocol["candidates"]:
        rows = [r for r in scores if r["checkpoint"] == candidate]
        if len(rows) != 55 or any(r["status"] != "complete" for r in rows):
            failed[candidate] = {"statuses": dict(Counter(r["status"] for r in rows)),
                                 "failures": [r for r in rows if r["status"] != "complete"]}
            continue
        comparisons.append(compare([*baseline, *rows], manifest)[0])
    vectors = {}
    qualify = []
    for item in comparisons:
        m = item["metrics"]
        vectors[item["checkpoint"]] = (
            m["chirality_all"]["edited_correct"], m["full_ez_all"]["edited_correct"],
            m["full_stereo_all"]["edited_correct"], -m["ideal_bond_rms"]["edited_median"],
            -m["ideal_angle_rms"]["edited_median"],
        )
        vectors[BASE] = (
            m["chirality_all"]["base_correct"], m["full_ez_all"]["base_correct"],
            m["full_stereo_all"]["base_correct"], -m["ideal_bond_rms"]["base_median"],
            -m["ideal_angle_rms"]["base_median"],
        )
        if (m["chirality_all"]["difference"] > 0 and m["full_stereo_all"]["difference"] > 0
                and m["full_ez_all"]["difference"] >= 0
                and m["ideal_bond_rms"]["edited_median"] < m["ideal_bond_rms"]["base_median"]
                and m["ideal_angle_rms"]["edited_median"] < m["ideal_angle_rms"]["base_median"]):
            qualify.append(item["checkpoint"])
    result = {
        "status": "development_comparison_complete", "comparisons": comparisons,
        "failed_evaluations": failed, "non_dominated_including_baseline": nondominated(vectors),
        "candidates_meeting_directional_criteria": qualify,
        "scores_sha256": digest(plan_path.with_suffix(".scores.json")),
        "protocol_sha256": digest(root / "protocol.json"),
        "interpretation": "This is candidate selection on development data, not an independent test. The original 209-system repair requirement remains unmet.",
    }
    write_json(root / "report.json", result)
    lines = ["# Matched comparison of all thirty candidates", "", result["interpretation"], "",
             "Every row uses the same 55 systems, baseline predictions, seed, and inference settings. Geometry values are median RMS errors against independent ideal conformers.", "",
             "| Candidate | R/S /55 | Full E/Z /55 | Joint /55 | Bond median (A) | Angle median (degrees) |",
             "|---|---:|---:|---:|---:|---:|"]
    order = [BASE, *protocol["candidates"]]
    for name in order:
        if name not in vectors:
            lines.append(f"| {name} | Evaluation failed | | | | |")
            continue
        rs, ez, joint, bond, angle = vectors[name]
        lines.append(f"| {name} | {rs} | {ez} | {joint} | {-bond:.6f} | {-angle:.4f} |")
    lines += ["", "## Predeclared directional criteria", "",
              "Higher R/S and joint agreement, nondecreasing full E/Z, and lower bond and angle medians, with all 55 evaluations complete.", "",
              *[f"- {name}" for name in qualify], "",
              "## Candidates without a strictly better alternative on all five endpoints", "",
              *[f"- {name}" for name in result["non_dominated_including_baseline"]], "",
              "Full confidence intervals, means, regressions, failures, and protein displacement outliers are retained in report.json. These descriptive intervals do not correct for selecting among thirty candidates.", ""]
    (root / "report.md").write_text("\n".join(lines))
    print(json.dumps({k: result[k] for k in ("status", "candidates_meeting_directional_criteria", "non_dominated_including_baseline")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "report"))
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.study)
    else:
        report(args.study)


if __name__ == "__main__":
    main()
