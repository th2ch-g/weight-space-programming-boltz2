"""Report a frozen comparison without treating repeated seeds as independent systems."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from summarize import compare

BASE = "boltz2_base.ckpt"
METRICS = ("chirality_all", "full_ez_all", "full_stereo_all", "ideal_bond_rms", "ideal_angle_rms")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def paired_seed_arrays(scores_by_seed, manifest, candidate):
    records = {r["id"]: r for r in manifest["records"] if r["split"] == "test"}
    ids = sorted(records)
    seeds = sorted(scores_by_seed)
    arrays = {metric: [] for metric in METRICS}
    for seed in seeds:
        rows = scores_by_seed[seed]
        keyed = {(r["id"], r["checkpoint"]): r for r in rows}
        expected = {(sid, checkpoint) for sid in ids for checkpoint in (BASE, candidate)}
        if len(keyed) != len(rows) or set(keyed) != expected:
            raise ValueError("Every seed must contain exactly the same paired test systems")
        for metric in METRICS:
            pair = []
            for checkpoint in (BASE, candidate):
                values = []
                for sid in ids:
                    row = keyed[sid, checkpoint]
                    if row["status"] != "complete" or row.get(metric) is None:
                        raise ValueError("Incomplete metrics cannot enter the pooled comparison")
                    values.append(row[metric])
                pair.append(values)
            arrays[metric].append(pair)
    arrays = {metric: np.asarray(values, dtype=float) for metric, values in arrays.items()}
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Nonfinite metrics cannot enter the pooled comparison")
    return ids, seeds, arrays


def pooled_test(scores_by_seed, manifest, candidate, draws=10000):
    ids, seeds, arrays = paired_seed_arrays(scores_by_seed, manifest, candidate)
    groups = {r["id"]: r["group"] for r in manifest["records"]}
    labels = sorted({groups[sid] for sid in ids})
    indices = [np.array([i for i, sid in enumerate(ids) if groups[sid] == label])
               for label in labels]
    counts = np.array([len(index) for index in indices])
    rng = np.random.default_rng(20261103)
    sampled = rng.integers(0, len(labels), size=(draws, len(labels)))
    result = {
        "systems": len(ids), "seeds": seeds, "sequence_scaffold_groups": len(labels),
        "definition": "Average each metric across the fixed seeds within each system, then compare systems. Bootstrap biological groups and retain every seed together.",
        "metrics": {},
    }
    for metric, values in arrays.items():
        base, edited = values.mean(axis=0)
        difference = edited - base
        sums = np.array([difference[index].sum() for index in indices])
        boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        record = {
            "base_mean": float(base.mean()), "edited_mean": float(edited.mean()),
            "paired_mean_change": float(difference.mean()),
            "paired_mean_change_cluster_95ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "base_median_of_seed_means": float(np.median(base)),
            "edited_median_of_seed_means": float(np.median(edited)),
            "systems_with_higher_edited_value": int((difference > 0).sum()),
            "systems_with_lower_edited_value": int((difference < 0).sum()),
        }
        if metric.endswith("_all"):
            record["base_correct_predictions"] = int(values[:, 0, :].sum())
            record["edited_correct_predictions"] = int(values[:, 1, :].sum())
            record["predictions_per_arm"] = len(ids) * len(seeds)
        result["metrics"][metric] = record
    return result


def integrity(plan, protocol, reused_runs):
    checks = []
    rows = []
    by_id = {}
    expected_settings = {
        "recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
        "fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False,
    }
    for task in plan["tasks"]:
        run_path = Path(task["output"]) / "run.json"
        if not run_path.is_file():
            rows.append({"id": task["id"], "checkpoint": task["checkpoint"], "status": "missing"})
            continue
        run = json.loads(run_path.read_text())
        rows.append({"id": task["id"], "checkpoint": task["checkpoint"], "status": run["status"]})
        if run["status"] != "complete":
            continue
        check = {
            "id": task["id"], "checkpoint": task["checkpoint"],
            "checkpoint_unchanged": run["checkpoint_sha256"] == protocol["checkpoint_sha256"][task["checkpoint"]],
            "seed": run["seed"] == task["seed"],
            "settings": all(run["settings"].get(k) == v for k, v in expected_settings.items()),
            "not_diagnostic": not run.get("diagnostic") and not run.get("reference_reflection"),
            "prediction_unchanged": digest(run_path.parent / run["prediction"]) == run["prediction_sha256"],
        }
        if task["output"] in reused_runs:
            check["reused_run_unchanged"] = digest(run_path) == reused_runs[task["output"]]
        else:
            check["frozen_runner"] = run["runner_sha256"] == protocol["source_sha256"]["scripts/predict.py"]
        checks.append(check)
        by_id.setdefault(task["id"], []).append(run)
    for sid, pair in by_id.items():
        if len(pair) != 2:
            continue
        for key in ("settings", "versions", "runner_sha256", "source_sha256", "gpu", "job_id", "input_sha256", "msa_sha256"):
            checks.append({"id": sid, "paired_field": key, "equal": pair[0].get(key) == pair[1].get(key)})
    ok = all(value is True for check in checks for key, value in check.items()
             if key not in {"id", "checkpoint", "paired_field"})
    return {"passed": ok, "statuses": dict(Counter(r["status"] for r in rows)), "checks": checks}


def report_markdown(report):
    lines = [
        "# Frozen Boltz-2 v2 evaluation", "",
        "The checkpoint was selected before opening the test split. Weights, inference settings, seeds, and endpoints were fixed in protocol.json.", "",
        "Development results informed selection. The earlier validation split also informed prior editing. Only the 104-system test split was previously unused.", "",
        "## Paired results", "",
        "Geometry values are medians across systems, relative to independent ETKDGv3/UFF conformers. They are not experimental binding-pose errors.", "",
        "| Population | Seed | Systems | R/S base -> v2 | Full E/Z base -> v2 | Joint base -> v2 | Bond median base -> v2 (A) | Angle median base -> v2 (degrees) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, comparison in report["comparisons"].items():
        metrics = comparison["metrics"]
        counts = [f"{metrics[k]['base_correct']} -> {metrics[k]['edited_correct']}"
                  for k in METRICS[:3]]
        bond, angle = metrics["ideal_bond_rms"], metrics["ideal_angle_rms"]
        population, seed = name.rsplit("_seed_", 1)
        lines.append(f"| {population} | {seed} | {comparison['complete_pairs']} | "
                     + " | ".join(counts)
                     + f" | {bond['base_median']:.6f} -> {bond['edited_median']:.6f}"
                     + f" | {angle['base_median']:.4f} -> {angle['edited_median']:.4f} |")
    lines += ["", "## Repeated-seed test comparison", "",
              report["pooled_test"]["definition"], "",
              "The intervals describe variation across the sampled biological groups, conditional on these three seeds. They do not establish robustness to arbitrary seeds.", "",
              "| Metric | Base mean | v2 mean | Paired difference | Cluster 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for metric, item in report["pooled_test"]["metrics"].items():
        low, high = item["paired_mean_change_cluster_95ci"]
        lines.append(f"| {metric} | {item['base_mean']:.8f} | {item['edited_mean']:.8f} | "
                     f"{item['paired_mean_change']:.8f} | [{low:.8f}, {high:.8f}] |")
    lines += ["", "Positive stereochemistry differences and negative geometry differences favor v2.",
              "", "## Original repair acceptance", "", "```json",
              json.dumps(report["original_acceptance"], indent=2), "```", "",
              "All per-seed confidence intervals, means, regressions, and protein displacement outliers are retained in report.json. Per-system scores and pair audits are retained beside the analysis plans.", "",
              "The original requirement remains zero R/S and E/Z errors on all 209 systems, with lower bond and angle medians and no missing or unverifiable systems. Completing this evaluation does not by itself satisfy that requirement.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    root = args.study
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    for name, checksum in protocol["source_sha256"].items():
        if digest(name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    for name, checksum in protocol["checkpoint_sha256"].items():
        if digest(name) != checksum:
            raise ValueError(f"Frozen checkpoint changed: {name}")
    plan_hashes = json.loads((root / "plan_manifest.json").read_text())
    for name, checksum in plan_hashes.items():
        if digest(name) != checksum:
            raise ValueError(f"Original plan changed: {name}")
    analysis_hashes = json.loads((root / "analysis_manifest.json").read_text())
    for name, checksum in analysis_hashes.items():
        if digest(name) != checksum:
            raise ValueError(f"Frozen analysis changed: {name}")
    names = ["development_seed_1", "validation_seed_1", "test_seed_1", "test_seed_2", "test_seed_3"]
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    plans = {name: Path(overrides.get(name, root / "plans" / f"{name}.json")) for name in names}
    integrity_reports = {}
    scores_by_name = {}
    comparisons = {}
    for name, path in plans.items():
        plan = json.loads(path.read_text())
        original = json.loads((root / "plans" / f"{name}.json").read_text())
        if sorted((t["id"], t["checkpoint"], t["seed"]) for t in plan["tasks"]) != sorted(
            (t["id"], t["checkpoint"], t["seed"]) for t in original["tasks"]
        ):
            raise ValueError("Retry plans must preserve every original system, checkpoint, and seed")
        result = integrity(plan, protocol, protocol["reused_run_sha256"])
        integrity_reports[name] = result
        if not result["passed"] or result["statuses"] != {"complete": len(plan["tasks"])}:
            write_json(root / "incomplete.json", integrity_reports)
            raise ValueError(f"Prediction or provenance incomplete: {name}")
        for script in ("evaluate.py", "audit_pairs.py"):
            subprocess.run([sys.executable, str(Path("scripts") / script), str(path)], check=True)
        score_path = path.with_suffix(".scores.json")
        rows = json.loads(score_path.read_text())
        if any(r["status"] != "complete" for r in rows):
            raise ValueError(f"Evaluation failed: {name}")
        scores_by_name[name] = rows
        comparisons[name] = compare(rows, manifest)[0]
    all_rows = sum((scores_by_name[name] for name in names[:3]), [])
    comparisons["all_209_seed_1"] = compare(all_rows, manifest)[0]
    write_json(root / "all_209_seed_1.scores.json", all_rows)
    test = pooled_test({seed: scores_by_name[f"test_seed_{seed}"] for seed in (1, 2, 3)},
                       manifest, protocol["candidate"])
    overall = comparisons["all_209_seed_1"]
    metrics = overall["metrics"]
    edited = [r for r in all_rows if r["checkpoint"] == protocol["candidate"]]
    gates = {
        "completed_209_systems": overall["complete_pairs"] == 209,
        "zero_rs_errors": metrics["chirality_all"]["edited_correct"] == 209,
        "zero_ez_errors": metrics["full_ez_all"]["edited_correct"] == 209,
        "lower_bond_median": metrics["ideal_bond_rms"]["edited_median"] < metrics["ideal_bond_rms"]["base_median"],
        "lower_angle_median": metrics["ideal_angle_rms"]["edited_median"] < metrics["ideal_angle_rms"]["base_median"],
        "no_unverifiable_ez_bonds": sum(r["unverifiable_ez_bonds"] for r in edited) == 0,
    }
    report = {
        "status": "evaluation_complete", "protocol_sha256": digest(root / "protocol.json"),
        "reporter_sha256": digest(Path(__file__)), "comparisons": comparisons,
        "pooled_test": test,
        "original_acceptance": {"passed": all(gates.values()), "checks": gates},
        "integrity": integrity_reports,
        "score_sha256": {name: digest(path.with_suffix(".scores.json")) for name, path in plans.items()},
        "center_counts": {name: {checkpoint: {
            key: sum(r[key] for r in rows if r["checkpoint"] == checkpoint)
            for key in ("chiral_centers", "chiral_correct", "full_ez_bonds", "full_ez_correct_bonds", "unverifiable_ez_bonds")
        } for checkpoint in (BASE, protocol["candidate"])} for name, rows in scores_by_name.items()},
        "systems_with_defined_chirality": {
            name: sum(r["chiral_centers"] > 0 for r in rows if r["checkpoint"] == BASE)
            for name, rows in scores_by_name.items()
        },
        "execution_retries": json.loads((root / "retries.json").read_text())
        if (root / "retries.json").exists() else [],
        "limitations": ["Development informed selection.", "Historical validation informed prior edits.",
                        "Test uncertainty is conditional on the three predeclared seeds.",
                        "The original model may have seen these complexes during training.",
                        "Geometry scores compare independent ideal conformers, not experimental ligand poses."],
    }
    write_json(root / "report.json", report)
    (root / "report.md").write_text(report_markdown(report))
    print(json.dumps({"status": report["status"], "original_acceptance": report["original_acceptance"],
                      "report": (root / "report.md").as_posix()}))


if __name__ == "__main__":
    main()
