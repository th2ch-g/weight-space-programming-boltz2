"""Rescore archived predictions with the RGI manuscript metrics of the public benchmark."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import rgi_benchmark

BASE = "boltz2_base.ckpt"
CANDIDATES = {
    "v2": "boltz2_wsp_reference_distances_v2.ckpt",
    "v11": "boltz2_wsp_absolute_vectors_only_v11.ckpt",
}
METRICS = ("match", "ez_match", "joint", "emb_brms", "emb_arms")
CALIBRATION_TOLERANCE = 1e-5
MOL_BUILD_SHA256 = "bc7463748663ee1eea21dc8e04d7abef5482466aa9e1952e812fe5658ade60f9"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    return Path(os.path.relpath(path)).as_posix()


def verify_evaluator(paper):
    import rgi_toolkit._mol_build as mol_build

    metadata = read_json(paper / "evaluation.json")
    hashes = rgi_benchmark.source_hashes()
    if digest(mol_build.__file__) != MOL_BUILD_SHA256:
        raise ValueError("Manuscript reference-preparation source differs")
    versions = {name: version(name) for name in metadata["software"]}
    if versions != metadata["software"]:
        raise ValueError(f"Manuscript software differs: {versions}")
    return {
        "benchmark": {"repository": rgi_benchmark.REPOSITORY, "commit": rgi_benchmark.COMMIT,
                      "source_sha256": hashes},
        "reference_preparation_sha256": MOL_BUILD_SHA256,
        "software": {**versions, "rgi-toolkit": version("rgi-toolkit")},
        "preparation": metadata["preparation"],
        "ez_method": metadata["ez_method"],
        "paper_evaluation_sha256": digest(paper / "evaluation.json"),
    }


def source_scores():
    yield "development", 1, Path("artifacts/all_candidates_development/plan.scores.json")
    for label in CANDIDATES:
        for split, seeds in (("validation", (1,)), ("test", (1, 2, 3))):
            for seed in seeds:
                yield label, seed, Path(
                    f"artifacts/frozen_{label}_evaluation/plans/{split}_seed_{seed}.scores.json"
                )


def prepare_tasks(manifest, paper, paper_predictions):
    records = {row["id"]: row for row in manifest["records"]}
    index_path = paper_predictions / "index_subct209.csv"
    with index_path.open() as handle:
        index = {row["sys_id"]: row for row in csv.DictReader(handle)}
    if index.keys() != records.keys():
        raise ValueError("Manuscript and WSP system membership differs")
    rows = {}
    for sid, row in records.items():
        if row["smiles"] != index[sid]["smiles"]:
            raise ValueError(f"Manuscript and WSP SMILES differ: {sid}")
        sdf = Path("data/ct209/sdf") / f"{sid}.sdf"
        if digest(sdf) != row["sdf_sha256"] or row["sdf_sha256"] != index[sid]["sdf_sha256"]:
            raise ValueError(f"Manuscript and WSP crystal reference differ: {sid}")
        rows[sid] = {**index[sid], "sdf_path": str(sdf)}
    jobs = {sid: {"id": sid, "paper": [], "wsp": []} for sid in records}
    arm_files = {"OFF": "off", "RGI": "on", "Native": "offsteer"}
    expected = {}
    for arm, suffix in arm_files.items():
        with (paper / f"resct209x_boltz2_{suffix}.csv").open() as handle:
            expected[arm] = {row["pdbid"]: row for row in csv.DictReader(handle)}
    with (paper / "evaluation_predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "boltz":
                jobs[row["system_id"]]["paper"].append({
                    "arm": row["arm"],
                    "prediction": str(paper_predictions / row["prediction"]),
                    "prediction_sha256": row["prediction_sha256"],
                    "expected": expected[row["arm"]][row["system_id"]],
                })
    source_hashes = {}
    pair_fields = (
        "seed", "settings", "versions", "runner_sha256", "source_sha256",
        "gpu", "job_id", "input_sha256", "msa_sha256",
    )
    for study, seed, path in source_scores():
        source_hashes[str(path)] = digest(path)
        saved = read_json(path)
        expected_split = "development" if study == "development" else path.name.split("_")[0]
        ids = {sid for sid, row in records.items() if row["split"] == expected_split}
        checkpoints = {row["checkpoint"] for row in saved}
        if study != "development" and checkpoints != {BASE, CANDIDATES[study]}:
            raise ValueError(f"Unexpected checkpoints: {path}")
        if study == "development" and (len(checkpoints) != 31 or BASE not in checkpoints):
            raise ValueError("Expected baseline and all thirty development candidates")
        keys = [(row["id"], row["checkpoint"]) for row in saved]
        if len(set(keys)) != len(keys) or set(keys) != {
            (sid, checkpoint) for sid in ids for checkpoint in checkpoints
        }:
            raise ValueError(f"Incomplete or duplicate source population: {path}")
        runs = {}
        for row in saved:
            run_path = Path(row["output"]) / "run.json"
            run = read_json(run_path)
            if row["status"] != "complete" or run["status"] != "complete":
                raise ValueError(f"Incomplete prediction: {run_path}")
            if run["seed"] != seed or run.get("diagnostic") or run.get("reference_reflection"):
                raise ValueError(f"Unexpected inference intervention: {run_path}")
            if any(run["settings"][key] for key in (
                "physical_guidance_update", "contact_guidance_update", "fk_steering",
            )):
                raise ValueError(f"Guidance enabled: {run_path}")
            task = {
                "id": row["id"], "checkpoint": row["checkpoint"], "study": study,
                "split": expected_split, "seed": seed,
                "prediction": str(run_path.parent / run["prediction"]),
                "prediction_sha256": run["prediction_sha256"],
                "run_sha256": digest(run_path),
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
            jobs[row["id"]]["wsp"].append(task)
            runs[row["id"], row["checkpoint"]] = run
        for (sid, checkpoint), run in runs.items():
            if any(run[key] != runs[sid, BASE][key] for key in pair_fields):
                raise ValueError(f"Unmatched inference conditions: {sid} {checkpoint}")
    if any(len(job["paper"]) != 3 for job in jobs.values()):
        raise ValueError("Calibration requires all three manuscript Boltz conditions")
    return rows, list(jobs.values()), source_hashes


def initialize_worker(rows):
    global ROWS
    ROWS = rows
    rgi_benchmark.load()


def score_prediction(path, sid):
    from evaluate import prediction_molecule, read_pdb

    row = ROWS[sid]
    metrics = rgi_benchmark.conformer_metrics(
        prediction_molecule(row["smiles"], read_pdb(Path(path))), row["sdf_path"]
    )
    result = {
        "match": metrics["chiral_match"], "ez_match": metrics["ez_match"],
        "emb_brms": metrics["ideal_bond_rms"], "emb_arms": metrics["ideal_angle_rms"],
        "bond_rms": metrics["crystal_bond_rms"], "angle_rms": metrics["crystal_angle_rms"],
        "ligand_rmsd": metrics["ligand_rmsd"],
    }
    for key in ("match", "ez_match", "emb_brms", "emb_arms"):
        if result[key] is None or not math.isfinite(result[key]):
            raise ValueError(f"Missing or nonfinite manuscript metric: {key}")
    result["joint"] = bool(result["match"] and result["ez_match"])
    return result


def score_system(job):
    calibration, scores, failures = [], [], []
    for task in job["paper"]:
        try:
            if digest(task["prediction"]) != task["prediction_sha256"]:
                raise ValueError("Manuscript prediction hash differs")
            result = score_prediction(task["prediction"], job["id"])
            mismatches = {}
            for key in ("match", "ez_match", "emb_brms", "emb_arms", "bond_rms", "angle_rms",
                        "ligand_rmsd"):
                saved = task["expected"][key]
                if key in ("match", "ez_match"):
                    differs = str(int(result[key])) != saved
                else:
                    differs = abs(result[key] - float(saved)) > CALIBRATION_TOLERANCE
                if differs:
                    mismatches[key] = {"saved": saved, "recomputed": result[key]}
            calibration.append({"id": job["id"], "arm": task["arm"], "mismatches": mismatches})
        except Exception as exc:
            failures.append({"id": job["id"], "paper_arm": task["arm"], "error": str(exc)})
    for task in job["wsp"]:
        try:
            if digest(task["prediction"]) != task["prediction_sha256"]:
                raise ValueError("WSP prediction hash differs")
            result = score_prediction(task["prediction"], job["id"])
            scores.append({**task, "status": "complete", **result})
        except Exception as exc:
            scores.append({**task, "status": "failed", "error": str(exc)})
            failures.append({"id": job["id"], "prediction": task["prediction"], "error": str(exc)})
    from rdkit import Chem

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ROWS[job["id"]]["smiles"]))
    try:
        reference = rgi_benchmark.load().ideal(smiles)[1].tolist()
    except ValueError:
        reference = None
    return {"id": job["id"], "calibration": calibration, "scores": scores,
            "failures": failures, "reference": {
                "canonical_smiles": smiles, "coordinates": reference,
            }}


def summary(rows):
    if not rows or any(row["status"] != "complete" for row in rows):
        raise ValueError("Incomplete scores cannot enter a summary")
    return {
        "systems": len(rows),
        **{key: sum(bool(row[key]) for row in rows) for key in METRICS[:3]},
        **{key: float(np.median([row[key] for row in rows])) for key in METRICS[3:]},
    }


def paired_effects(rows, records, candidate, seeds):
    keyed = {(row["id"], row["checkpoint"], row["seed"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate system/checkpoint/seed")
    ids = sorted({row["id"] for row in rows})
    expected = {(sid, checkpoint, seed) for sid in ids
                for checkpoint in (BASE, candidate) for seed in seeds}
    if set(keyed) != expected or any(row["status"] != "complete" for row in rows):
        raise ValueError("Paired effects require complete, identical seed membership")
    labels = sorted({records[sid]["group"] for sid in ids})
    indices = [np.array([i for i, sid in enumerate(ids) if records[sid]["group"] == label])
               for label in labels]
    counts = np.array([len(index) for index in indices])
    draws = np.random.default_rng(20261103).integers(0, len(labels), (10000, len(labels)))
    metrics = {}
    for metric in METRICS:
        values = np.array([[[keyed[sid, checkpoint, seed][metric] for sid in ids]
                            for checkpoint in (BASE, candidate)] for seed in seeds], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite paired metric")
        baseline, edited = values.mean(axis=0)
        change = edited - baseline
        group_changes = np.array([change[index].sum() for index in indices])
        boot = group_changes[draws].sum(axis=1) / counts[draws].sum(axis=1)
        metrics[metric] = {
            "base_mean": float(baseline.mean()), "edited_mean": float(edited.mean()),
            "paired_mean_change": float(change.mean()),
            "cluster_95ci": np.percentile(boot, [2.5, 97.5]).tolist(),
        }
        if metric in METRICS[:3]:
            metrics[metric].update(base_correct=int(values[:, 0].sum()),
                                   edited_correct=int(values[:, 1].sum()),
                                   predictions_per_arm=len(ids) * len(seeds))
    return {"systems": len(ids), "seeds": seeds, "groups": len(labels), "metrics": metrics}


def make_report(scores, manifest):
    records = {row["id"]: row for row in manifest["records"]}
    development = [row for row in scores if row["study"] == "development"]
    summaries = {"development": {checkpoint: summary([
        row for row in development if row["checkpoint"] == checkpoint
    ]) for checkpoint in sorted({row["checkpoint"] for row in development})}}
    effects = {}
    for label, candidate in CANDIDATES.items():
        rows = [row for row in scores if row["study"] == label]
        for split, seeds in (("validation", (1,)), ("test", (1, 2, 3))):
            for seed in seeds:
                subset = [row for row in rows if row["split"] == split and row["seed"] == seed]
                summaries[f"{label}_{split}_seed_{seed}"] = {
                    checkpoint: summary([row for row in subset if row["checkpoint"] == checkpoint])
                    for checkpoint in (BASE, candidate)
                }
        nondevelopment = [row for row in rows if row["seed"] == 1]
        summaries[f"{label}_nondevelopment_seed_1"] = {
            checkpoint: summary([row for row in nondevelopment if row["checkpoint"] == checkpoint])
            for checkpoint in (BASE, candidate)
        }
        effects[label] = {
            "nondevelopment_seed_1": paired_effects(nondevelopment, records, candidate, [1]),
            "test_seeds_1_2_3": paired_effects(
                [row for row in rows if row["split"] == "test"], records, candidate, [1, 2, 3]
            ),
        }
    b2 = {(row["id"], row["seed"]): row for row in scores
          if row["study"] == "v2" and row["checkpoint"] == BASE}
    b11 = {(row["id"], row["seed"]): row for row in scores
           if row["study"] == "v11" and row["checkpoint"] == BASE}
    if b2.keys() != b11.keys() or any(
        b2[key][field] != b11[key][field] for key in b2 for field in (*METRICS, "prediction_sha256")
    ):
        raise ValueError("The two frozen studies have different baselines")
    return {"summaries": summaries, "paired_effects": effects,
            "identical_frozen_baselines": True}


def render_report(report):
    lines = ["# Re-evaluation with the RGI manuscript method", "",
             "Archived predictions were rescored; no model inference or weight editing was performed.",
             "R/S and E/Z compare input-graph CIP labels with labels perceived from predicted",
             "coordinates, using the public RGI-Toolkit benchmark. E/Z includes only acyclic,",
             "nonaromatic, heavy-atom-decidable defined bonds. Geometry uses the manuscript's",
             "independent canonical-SMILES ETKDGv3/UFF/stereo-validated reference preparation.", "",
             "The joint count is the conjunction of the manuscript's two per-system classifications.",
             "It is an additional summary, not the original common-mapping full-stereochemistry gate.",
             "The existing 55/50/104 split and previous validation/test exposure are unchanged.",
             "The original full-stereochemistry repair requirement remains unmet.", "",
             "All geometry entries below are medians of per-system RMS errors.", ""]
    for name, arms in report["summaries"].items():
        lines += [f"## {name}", "", "| Checkpoint | N | R/S | E/Z | Joint | Bond (A) | Angle (degrees) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for checkpoint, row in arms.items():
            lines.append(f"| {checkpoint} | {row['systems']} | {row['match']} | {row['ez_match']} | "
                         f"{row['joint']} | {row['emb_brms']:.6f} | {row['emb_arms']:.4f} |")
        lines.append("")
    lines += ["## Paired effects", "",
              "Average repeated seeds within each system, then bootstrap sequence/scaffold groups",
              "10,000 times with resampling seed 20261103. These intervals are conditional on the",
              "evaluated seeds and do not correct for candidate selection.", ""]
    for label, populations in report["paired_effects"].items():
        for population, result in populations.items():
            lines += [f"### {label}: {population}", "", "| Metric | Mean change | Cluster 95% CI |",
                      "|---|---:|---|"]
            for metric, row in result["metrics"].items():
                low, high = row["cluster_95ci"]
                lines.append(f"| {metric} | {row['paired_mean_change']:.8f} | [{low:.8f}, {high:.8f}] |")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-data", type=Path, required=True)
    parser.add_argument("--paper-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CPU workers")
    output = args.output
    if (output / "protocol.json").exists():
        raise FileExistsError("Use a new evaluation output directory")
    evaluator = verify_evaluator(args.paper_data)
    manifest_path = Path("data/ct209/manifest.json")
    manifest = read_json(manifest_path)
    rows, jobs, source_hashes = prepare_tasks(manifest, args.paper_data, args.paper_predictions)
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "Rescore saved WSP predictions using the RGI manuscript metrics.",
        "evaluation_only": True, "original_repair_gate_unchanged": True,
        "manifest_sha256": digest(manifest_path), "source_score_sha256": source_hashes,
        "evaluator": evaluator, "driver_sha256": digest(__file__),
        "split_counts": manifest["counts"], "cpu_workers": args.workers,
        "planned_predictions": sum(len(job["wsp"]) for job in jobs),
        "planned_calibration_predictions": sum(len(job["paper"]) for job in jobs),
        "inputs": {key: relative(value) for key, value in {
            "paper_data": args.paper_data,
            "paper_predictions": args.paper_predictions,
        }.items()},
    }
    write_json(output / "protocol.json", protocol)
    (output / "source" / "rgi_benchmark").mkdir(parents=True)
    for name in rgi_benchmark.SOURCE_SHA256:
        shutil.copyfile(rgi_benchmark.directory() / name, output / "source" / "rgi_benchmark" / name)
    for path in (Path(__file__), Path(rgi_benchmark.__file__), Path(__file__).with_name("evaluate.py")):
        shutil.copyfile(path, output / "source" / path.name)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(rows,)) as pool:
        for result in pool.map(score_system, jobs):
            results.append(result)
            write_json(output / "systems" / f"{result['id']}.json", result)
            if len(results) % 10 == 0 or len(results) == len(jobs):
                print(json.dumps({"systems_complete": len(results), "systems_total": len(jobs),
                                  "failures": sum(len(item['failures']) for item in results)}), flush=True)
    scores = [row for result in results for row in result["scores"]]
    calibration = [row for result in results for row in result["calibration"]]
    failures = [row for result in results for row in result["failures"]]
    mismatches = [row for row in calibration if row["mismatches"]]
    write_json(output / "scores.json", scores)
    write_json(output / "calibration.json", {"predictions": len(calibration),
               "matched": len(calibration) - len(mismatches), "mismatches": mismatches})
    write_json(output / "failures.json", failures)
    if failures or mismatches or len(scores) != protocol["planned_predictions"]:
        raise ValueError("Evaluation or manuscript calibration failed; inspect retained results")
    if any(digest(path) != checksum for path, checksum in source_hashes.items()):
        raise ValueError("Archived source scores changed during evaluation")
    verify_evaluator(args.paper_data)
    report = {"status": "complete", "predictions": len(scores),
              "counts": dict(Counter(row["study"] for row in scores)),
              "manuscript_calibration_predictions": len(calibration),
              "manuscript_calibration_mismatches": 0,
              "protocol_sha256": digest(output / "protocol.json"),
              "scores_sha256": digest(output / "scores.json"),
              **make_report(scores, manifest)}
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_report(report))
    print(json.dumps({"status": "complete", "predictions": len(scores),
                      "calibration_predictions": len(calibration), "failures": 0}), flush=True)


if __name__ == "__main__":
    main()
