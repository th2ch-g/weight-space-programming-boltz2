"""Stage the exact edited checkpoints and their public provenance for the Hub."""

import argparse
import hashlib
import json
import os
import re
import shutil
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
from evaluate_paper import BASE, CANDIDATES, summary

PAPER_EVALUATION = Path("artifacts/paper_evaluation")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def paper_pooled_summary(scores, ids):
    """Summarize development and v11 validation/test predictions at seed 1."""
    pooled = {}
    for checkpoint in (BASE, CANDIDATES["v11"]):
        rows = [row for row in scores if row["checkpoint"] == checkpoint and row["seed"] == 1
                and row["study"] in ("development", "v11")]
        if sorted(row["id"] for row in rows) != sorted(ids):
            raise ValueError("The pooled population must contain each system exactly once")
        pooled[checkpoint] = {
            **summary(rows),
            "emb_brms_mean": float(np.mean([row["emb_brms"] for row in rows])),
            "emb_arms_mean": float(np.mean([row["emb_arms"] for row in rows])),
        }
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--upstream-revision", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_absolute():
        raise ValueError("Use a new relative release directory")
    candidates = sorted(Path.cwd().glob("boltz2_wsp_*.ckpt"),
                        key=lambda p: int(re.search(r"_v(\d+)\.ckpt$", p.name)[1]))
    if [int(re.search(r"_v(\d+)\.ckpt$", p.name)[1]) for p in candidates] != list(range(1, 31)):
        raise ValueError("Expected all thirty edited candidates")
    base = json.loads(Path("artifacts/base_provenance.json").read_text())["sha256"]
    report = json.loads(Path("artifacts/all_candidates_development/report.json").read_text())
    if len(report["comparisons"]) != 30 or report["failed_evaluations"]:
        raise ValueError("The development comparison is incomplete")
    comparisons = {r["checkpoint"]: r for r in report["comparisons"]}
    paper_report = json.loads((PAPER_EVALUATION / "report.json").read_text())
    if (paper_report["status"] != "complete"
            or digest(PAPER_EVALUATION / "scores.json") != paper_report["scores_sha256"]):
        raise ValueError("The manuscript-evaluator rescoring is incomplete or changed")
    manifest_ids = [row["id"] for row in json.loads(
        Path("data/ct209/manifest.json").read_text())["records"]]
    pooled = paper_pooled_summary(
        json.loads((PAPER_EVALUATION / "scores.json").read_text()), manifest_ids
    )
    args.output.mkdir(parents=True)
    (args.output / "edits").mkdir()
    (args.output / "evaluation").mkdir()
    entries = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release checkpoints must be regular files")
        record = json.loads(Path("artifacts", path.stem + ".edit.json").read_text())
        checksum = digest(path)
        if checksum != record["output_sha256"] or record["base_sha256"] != base:
            raise ValueError(f"Checkpoint provenance mismatch: {path.name}")
        if not record["all_untargeted_tensors_equal"]:
            raise ValueError("Untargeted tensors changed")
        os.link(path, args.output / path.name)
        spec = Path("edits", path.stem.removeprefix("boltz2_wsp_") + ".json")
        shutil.copyfile(spec, args.output / "edits" / spec.name)
        metrics = comparisons[path.name]["metrics"]
        entries.append({
            "candidate": f"v{int(re.search(r'_v(\d+)\.ckpt$', path.name)[1])}",
            "file": path.name, "sha256": checksum, "bytes": path.stat().st_size,
            "base_sha256": base, "edit_spec": f"edits/{spec.name}",
            "changed_elements": record["changed_elements"],
            "total_elements": record["total_elements"],
            "status": "experimental_unvalidated",
            "development_55_seed_1": {
                "rs_correct_systems": metrics["chirality_all"]["edited_correct"],
                "full_ez_correct_systems": metrics["full_ez_all"]["edited_correct"],
                "joint_correct_systems": metrics["full_stereo_all"]["edited_correct"],
                "median_bond_rms_angstrom": metrics["ideal_bond_rms"]["edited_median"],
                "median_angle_rms_degrees": metrics["ideal_angle_rms"]["edited_median"],
            },
        })
        print(json.dumps({"verified": path.name, "sha256": checksum}), flush=True)
    upstream = f"https://huggingface.co/boltz-community/boltz-2/resolve/{args.upstream_revision}/boltz2_conf.ckpt"
    manifest = {
        "format": "pytorch-lightning-checkpoint", "boltz_version": "2.2.1",
        "github": args.github_url, "model_id": args.model_id,
        "upstream_checkpoint": upstream, "upstream_sha256": base,
        "checkpoint_count": len(entries), "total_logical_bytes": sum(r["bytes"] for r in entries),
        "checkpoints": entries,
        "interpretation": "All files are experimental candidates. No candidate has met the original 209-system zero-error and geometry-improvement acceptance criteria.",
    }
    (args.output / "CHECKPOINTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{r['sha256']}  {r['file']}\n" for r in entries)
    )
    license_path = distribution("boltz").locate_file("boltz-2.2.1.dist-info/licenses/LICENSE")
    license_text = Path(license_path).read_text()
    if "MIT License" not in license_text or "Jeremy Wohlwend" not in license_text:
        raise ValueError("Unexpected upstream license")
    (args.output / "LICENSE").write_text(license_text)
    for name in ["all_candidates_development", "frozen_v2_evaluation", "frozen_v11_evaluation"]:
        source = Path("artifacts") / name / "report.md"
        if source.is_file():
            shutil.copyfile(source, args.output / "evaluation" / f"{name}.md")
    shutil.copyfile(
        "artifacts/all_candidates_development/reproducibility_check.json",
        args.output / "evaluation" / "reproducibility_check.json",
    )
    shutil.copyfile(PAPER_EVALUATION / "report.md", args.output / "evaluation" / "paper_evaluation.md")
    (args.output / "evaluation" / "paper_evaluation_v11_209.json").write_text(json.dumps({
        "population": "55 development systems plus v11 validation and test systems at seed 1",
        "includes_selection_systems": True,
        "scores_sha256": paper_report["scores_sha256"],
        "summaries": pooled,
    }, indent=2) + "\n")
    lines = [
        "---", "license: mit", "base_model: boltz-community/boltz-2",
        "datasets:", "- wasarou/plinder-cistrans-209",
        "tags:", "- biology", "- protein-structure-prediction", "- boltz2",
        "- weight-space-programming", "- experimental", "---", "",
        "# Experimental weight-space edits to Boltz-2", "",
        "This repository contains the exact thirty edited Lightning checkpoints (v1-v30) from a direct weight-edit experiment with Boltz-2 2.2.1. These are unvalidated research checkpoints. No checkpoint meets the original requirement of zero R/S and E/Z errors across all 209 systems with lower bond and angle errors.", "",
        f"[Source code]({args.github_url}) | [Checkpoint manifest](CHECKPOINTS.json) | [SHA-256 checksums](SHA256SUMS)", "",
        "## Provenance", "",
        f"The unmodified baseline is the [upstream Boltz-2 structure checkpoint]({upstream}). Its SHA-256 is `{base}`. The baseline is not duplicated here. Each edited file was checked against its original edit record, and all untargeted tensors were verified during checkpoint creation. The files retain the upstream MIT license and copyright notice in LICENSE.", "",
        "## Evaluation scope", "",
        "All thirty candidates were compared on the same 55 development systems at seed 1, with a shared baseline and matched GPU allocations. No candidate met the simultaneous-improvement criteria. v11 had better development point estimates than v2 for R/S, joint stereochemistry, and the two geometry medians, but its bond-length median was worse than the baseline under the repository evaluator. These development results do not establish a validated replacement.", "",
        "The frozen v2 follow-up evaluated 50 validation systems at seed 1 and 104 test systems at seeds 1, 2, and 3. Joint test agreement was unchanged at 103/312 predictions. All primary pooled-effect confidence intervals included zero. Repeated seeds were averaged within each system before biological-group resampling.", "",
        "The v11 validation/test follow-up is recorded separately. The same test split had already been used for v2, so it is not a new untouched test.", "",
        "- [Full thirty-candidate development comparison](evaluation/all_candidates_development.md)",
        "- [Frozen v2 evaluation](evaluation/frozen_v2_evaluation.md)",
    ]
    if (args.output / "evaluation/frozen_v11_evaluation.md").is_file():
        lines.append("- [Frozen v11 evaluation](evaluation/frozen_v11_evaluation.md)")
    else:
        lines.append("- The v11 validation/test evaluation is still running; its completed report will be added when available.")
    base_pool, v11_pool = pooled[BASE], pooled[CANDIDATES["v11"]]
    lines += ["- [RGI manuscript evaluator rescoring](evaluation/paper_evaluation.md)", "",
              "## RGI manuscript evaluator", "",
              "The archived predictions were also rescored with the RGI-Toolkit manuscript evaluator, which reproduced all 627 manuscript Boltz-2 predictions. It uses a different reference conformer from the repository evaluator and reverses the sign of the small v11 development bond-length difference. The table pools the 55 development systems with the v11 validation and test systems at seed 1. This pool includes the systems used for selection. All paired confidence intervals for the non-development and test populations included zero.", "",
              "| Checkpoint | R/S | E/Z | Joint | Bond median (A) | Angle median (degrees) | Bond mean (A) | Angle mean (degrees) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in (("Upstream baseline", base_pool), ("v11", v11_pool)):
        lines.append(f"| {label} | {row['match']}/{row['systems']} | {row['ez_match']}/{row['systems']} | "
                     f"{row['joint']}/{row['systems']} | {row['emb_brms']:.6f} | {row['emb_arms']:.4f} | "
                     f"{row['emb_brms_mean']:.6f} | {row['emb_arms_mean']:.4f} |")
    lines += ["", "## Development results", "",
              "Counts are complete-system agreements out of 55. Geometry values are median RMS errors against independent ETKDGv3/UFF conformers, not experimental binding-pose errors.", "",
              "| Candidate | R/S | Full E/Z | Joint | Bond median (A) | Angle median (degrees) |",
              "|---|---:|---:|---:|---:|---:|"]
    baseline = report["comparisons"][0]["metrics"]
    lines.append(
        f"| Upstream baseline | {baseline['chirality_all']['base_correct']}/55 | "
        f"{baseline['full_ez_all']['base_correct']}/55 | {baseline['full_stereo_all']['base_correct']}/55 | "
        f"{baseline['ideal_bond_rms']['base_median']:.6f} | {baseline['ideal_angle_rms']['base_median']:.4f} |"
    )
    for row in entries:
        m = row["development_55_seed_1"]
        lines.append(f"| [{row['candidate']}]({row['file']}) | {m['rs_correct_systems']}/55 | "
                     f"{m['full_ez_correct_systems']}/55 | {m['joint_correct_systems']}/55 | "
                     f"{m['median_bond_rms_angstrom']:.6f} | {m['median_angle_rms_degrees']:.4f} |")
    lines += ["", "## Download", "", "```bash",
              "uv tool install huggingface-hub",
              f"hf download {args.model_id} boltz2_wsp_absolute_vectors_only_v11.ckpt --local-dir .",
              "```", "",
              "Use the checkpoints with Boltz-2 2.2.1. The source repository documents the experiment and its inference runner. The reported paired comparisons disable physical, contact, and FK guidance; standard upstream CLI defaults are not necessarily benchmark-equivalent.", "",
              "Prepared inputs, full predictions, and private session transcripts are not distributed in this model repository. The public dataset is linked in the model-card metadata; reproducing the archived campaign also requires the prepared benchmark inputs and legacy evaluation dependencies described in the source repository.", "",
              "## References", "",
              "- Passaro et al. (2025). [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707).",
              "- Takehi et al. (2026). [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).", ""]
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"stage": str(args.output), "checkpoints": len(entries),
                      "total_logical_bytes": manifest["total_logical_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
