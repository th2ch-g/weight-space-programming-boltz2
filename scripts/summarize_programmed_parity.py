import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    expected = {row["id"] for row in manifest["records"] if row["split"] == "development"}
    systems = {}
    hashes = set()
    sources = {}
    execution_failures = []
    for path in args.reports:
        report = json.loads(path.read_text())
        source_hash = report.get("program_source_sha256", report.get("source_sha256"))
        if not source_hash:
            raise ValueError(f"Missing program source identity: {path}")
        hashes.add(source_hash)
        sources[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if report.get("status") == "partial_execution_failed":
            execution_failures.append({key: report[key] for key in ("failed_job", "failure", "source_log", "source_log_sha256")})
        for row in report["systems"]:
            if row["id"] in systems or row["id"] not in expected:
                raise ValueError("Repeated or non-development system in diagnostic reports")
            if sorted(precision["bfloat16_autocast"] for precision in row["precisions"]) != [False, True]:
                raise ValueError("Each system requires float32 and bfloat16 observations")
            systems[row["id"]] = row
    if len(hashes) != 1:
        raise ValueError("The reports use different programmed module sources")
    if set(systems) != expected:
        raise ValueError(f"Missing development systems: {sorted(expected - systems.keys())}")
    summaries = []
    for mixed in (False, True):
        centers = []
        for sid, row in sorted(systems.items()):
            precision = next(p for p in row["precisions"] if p["bfloat16_autocast"] == mixed)
            centers.extend({"id": sid, **center} for center in precision["centers"])
        def passed(center):
            return all(center[key] == 0 for key in (
                "proper_rotation_sign_errors", "reflected_rotation_sign_errors", "translation_sign_errors"
            )) and center["minimum_absolute_selected_feature"] > 0
        failed = [center for center in centers if not passed(center)]
        summaries.append({
            "bfloat16_autocast": mixed,
            "systems": len(systems),
            "centers": len(centers),
            "all_checks_passed_centers": len(centers) - len(failed),
            "all_checks_passed_systems": len(systems) - len({center["id"] for center in failed}),
            "proper_rotation_sign_errors": sum(c["proper_rotation_sign_errors"] for c in centers),
            "proper_rotation_observations": 24 * len(centers),
            "reflected_rotation_sign_errors": sum(c["reflected_rotation_sign_errors"] for c in centers),
            "reflected_rotation_observations": 24 * len(centers),
            "translation_sign_errors": sum(c["translation_sign_errors"] for c in centers),
            "translation_observations": 2 * len(centers),
            "minimum_absolute_selected_feature": min(c["minimum_absolute_selected_feature"] for c in centers),
            "failed_centers": failed,
        })
    result = {
        "scope": "Development-only parity-feature diagnostics, not predicted stereochemistry or a repair acceptance result.",
        "program_source_sha256": next(iter(hashes)),
        "sources": sources,
        "execution_failures_retained": execution_failures,
        "completed_systems": len(systems),
        "test_predictions_used": False,
        "summaries": summaries,
        "limitation": "The tested encoder and Pairformer weights were assigned analytically and replace the pretrained modules. The largest reference component is selected outside the model for diagnosis. These results do not establish a usable checkpoint, correct R/S labels, E/Z repair, improved bond geometry, or preserved protein predictions.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}))


if __name__ == "__main__":
    main()
