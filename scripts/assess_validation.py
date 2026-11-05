import argparse
import json
from pathlib import Path

from summarize import compare, digest

parser = argparse.ArgumentParser()
parser.add_argument("--scores", type=Path, default=Path("artifacts/validation_v1.scores.json"))
args = parser.parse_args()
selection_path = Path("artifacts/validation_candidate.json")
selection = json.loads(selection_path.read_text())
manifest = json.loads(Path("data/ct209/manifest.json").read_text())
scores = json.loads(args.scores.read_text())
report = compare(scores, manifest)[0]
assert report["checkpoint"] == selection["checkpoint"]
metrics = report["metrics"]
checks = {
    "all_pairs_completed": report["complete_pairs"] == report["planned_pairs"] == 50,
    "same_gpu_within_each_pair": report["all_pairs_same_gpu_type"],
    "joint_stereo_improves": metrics["stereo_all"]["edited_correct"]
    > metrics["stereo_all"]["base_correct"],
    "ez_does_not_decrease": metrics["ez_all"]["edited_correct"]
    >= metrics["ez_all"]["base_correct"],
    "bond_error_within_limit": metrics["ideal_bond_rms"]["edited_median"]
    <= 1.10 * metrics["ideal_bond_rms"]["base_median"],
    "angle_error_within_limit": metrics["ideal_angle_rms"]["edited_median"]
    <= 1.10 * metrics["ideal_angle_rms"]["base_median"],
    "median_protein_change_within_limit": metrics["paired_protein_ca_rmsd"]["median"] < 0.5,
}
output = {
    "status": "accepted_for_test" if all(checks.values()) else "validation_failed",
    "checks": checks,
    "comparison": report,
    "selection_sha256": digest(selection_path),
    "scores_sha256": digest(args.scores),
}
Path("artifacts/validation_assessment.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output))
