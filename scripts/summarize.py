import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_difference_interval(base, edited, group_indices, draws):
    values = []
    for draw in draws:
        indices = np.concatenate([group_indices[index] for index in draw])
        if len(indices):
            values.append(float(np.median(edited[indices]) - np.median(base[indices])))
    if not values:
        raise ValueError("No valid paired median bootstrap samples")
    return np.percentile(values, [2.5, 97.5]).tolist()


def compare(rows, manifest):
    grouped = {}
    for row in rows:
        key = (row["id"], row["checkpoint"])
        if key in grouped:
            raise ValueError(f"Duplicate prediction: {key}")
        grouped[key] = row
    base = {
        sid: row for (sid, checkpoint), row in grouped.items() if checkpoint == "boltz2_base.ckpt"
    }
    groups = {row["id"]: row["group"] for row in manifest["records"]}
    reports = []
    for checkpoint in sorted({r["checkpoint"] for r in rows} - {"boltz2_base.ckpt"}):
        edited = {sid: row for (sid, ckpt), row in grouped.items() if ckpt == checkpoint}
        ids = sorted(set(base) & set(edited))
        complete = [
            sid for sid in ids if base[sid]["status"] == edited[sid]["status"] == "complete"
        ]
        if not complete:
            raise ValueError("No paired completed predictions")
        cluster_ids = sorted({groups[sid] for sid in complete})
        counts = np.array([sum(groups[sid] == group for sid in complete) for group in cluster_ids])
        rng = np.random.default_rng(20261103)
        draws = rng.integers(0, len(cluster_ids), size=(10000, len(cluster_ids)))
        summary = {
            "checkpoint": checkpoint,
            "planned_pairs": len(ids),
            "complete_pairs": len(complete),
            "incomplete_ids": sorted(set(ids) - set(complete)),
            "clusters": len(cluster_ids),
            "metrics": {},
        }
        metrics = ["chirality_all", "benchmark_chirality_all", "ez_all", "stereo_all"]
        metrics += [
            name for name in ("full_ez_all", "full_stereo_all") if any(name in row for row in rows)
        ]
        for metric in metrics:
            if any(
                base[sid].get(metric) is None or edited[sid].get(metric) is None for sid in complete
            ):
                raise ValueError(f"Missing primary metric {metric}")
            b = np.array([base[sid][metric] for sid in complete], dtype=int)
            e = np.array([edited[sid][metric] for sid in complete], dtype=int)
            difference = e - b
            improved, worsened = int((difference > 0).sum()), int((difference < 0).sum())
            cluster_changes = np.array(
                [
                    sum(difference[i] for i, sid in enumerate(complete) if groups[sid] == group)
                    for group in cluster_ids
                ]
            )
            boot = cluster_changes[draws].sum(axis=1) / counts[draws].sum(axis=1)
            summary["metrics"][metric] = {
                "base_correct": int(b.sum()),
                "edited_correct": int(e.sum()),
                "n": len(complete),
                "improved": improved,
                "worsened": worsened,
                "difference": float(difference.mean()),
                "cluster_bootstrap_95ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                "unclustered_discordant_binomial_p": float(
                    binomtest(improved, improved + worsened).pvalue
                )
                if improved + worsened
                else 1.0,
            }
        for metric in (
            "ideal_bond_rms",
            "ideal_angle_rms",
            "inter_clash_min_relative",
            "inter_clash_pairs",
            "intra_clash_pairs",
        ):
            valid = [
                sid
                for sid in complete
                if base[sid].get(metric) is not None and edited[sid].get(metric) is not None
            ]
            summary["metrics"][metric] = (
                {
                    "n": len(valid),
                    "base_median": float(np.median([base[sid][metric] for sid in valid])),
                    "edited_median": float(np.median([edited[sid][metric] for sid in valid])),
                }
                if valid
                else {"n": 0}
            )
            if valid and metric in ("ideal_bond_rms", "ideal_angle_rms"):
                b = np.array([base[sid][metric] for sid in valid])
                e = np.array([edited[sid][metric] for sid in valid])
                if not np.isfinite(b).all() or not np.isfinite(e).all():
                    raise ValueError(f"Nonfinite geometry errors in {metric}")
                difference = e - b
                group_indices = [
                    np.array([i for i, sid in enumerate(valid) if groups[sid] == group], dtype=int)
                    for group in cluster_ids
                ]
                group_counts = np.array(
                    [sum(groups[sid] == group for sid in valid) for group in cluster_ids]
                )
                changes = np.array(
                    [
                        sum(difference[i] for i, sid in enumerate(valid) if groups[sid] == group)
                        for group in cluster_ids
                    ]
                )
                denominators = group_counts[draws].sum(axis=1)
                boot = changes[draws].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
                summary["metrics"][metric].update(
                    base_mean=float(b.mean()),
                    edited_mean=float(e.mean()),
                    improved=int((difference < 0).sum()),
                    worsened=int((difference > 0).sum()),
                    paired_mean_change=float(difference.mean()),
                    paired_mean_change_cluster_95ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                    difference_of_medians=float(np.median(e) - np.median(b)),
                    difference_of_medians_cluster_95ci=median_difference_interval(
                        b, e, group_indices, draws
                    ),
                    median_paired_change=float(np.median(difference)),
                )
        for metric in ("paired_protein_ca_rmsd", "paired_ligand_rmsd_after_protein_fit"):
            values = [
                edited[sid][metric] for sid in complete if edited[sid].get(metric) is not None
            ]
            summary["metrics"][metric] = (
                {
                    "n": len(values),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(max(values)),
                }
                if values
                else {"n": 0}
            )
        summary["all_pairs_same_gpu_type"] = all(
            edited[sid].get("paired_gpu_equal") is True for sid in complete
        )
        summary["protein_outlier_threshold_angstrom"] = 0.5
        summary["protein_outliers"] = [
            {"id": sid, "paired_protein_ca_rmsd": edited[sid]["paired_protein_ca_rmsd"]}
            for sid in complete
            if edited[sid].get("paired_protein_ca_rmsd", 0) > 0.5
        ]
        reports.append(summary)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    rows = [row for path in args.scores for row in json.loads(path.read_text())]
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    reports = compare(rows, manifest)
    payload = {
        "stage": args.stage,
        "sources": {p.as_posix(): digest(p) for p in args.scores},
        "split_manifest_sha256": digest(Path("data/ct209/manifest.json")),
        "comparisons": reports,
        "uncertainty_method": "10000 paired bootstrap resamples of sequence/scaffold connected components; seed 20261103",
        "interpretation": "Development and validation results inform editing or selection and are not independent tests. Single-seed comparisons do not establish seed robustness.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"# {args.stage} comparison", "", payload["interpretation"], ""]
    for report in reports:
        lines += [
            f"## {report['checkpoint']}",
            "",
            f"Completed pairs: {report['complete_pairs']}/{report['planned_pairs']}; independent split groups: {report['clusters']}.",
            "",
            "| Metric | Base | Edited | Improved | Worsened | Difference, percentage points (cluster 95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        stereo_metrics = ["chirality_all", "ez_all", "stereo_all"]
        stereo_metrics += [
            name for name in ("full_ez_all", "full_stereo_all") if name in report["metrics"]
        ]
        for metric in stereo_metrics:
            item = report["metrics"][metric]
            low, high = item["cluster_bootstrap_95ci"]
            lines.append(
                f"| {metric} | {item['base_correct']}/{item['n']} | {item['edited_correct']}/{item['n']} | {item['improved']} | {item['worsened']} | {100 * item['difference']:.2f} ({100 * low:.2f}, {100 * high:.2f}) |"
            )
        lines += [
            "",
            "```json",
            json.dumps(
                {
                    k: v
                    for k, v in report["metrics"].items()
                    if k not in {*stereo_metrics, "benchmark_chirality_all"}
                },
                indent=2,
            ),
            "```",
            "",
        ]
        if report["protein_outliers"]:
            lines += [
                "Protein CA displacement exceeds 0.5 A in these systems:",
                "",
                "```json",
                json.dumps(report["protein_outliers"], indent=2),
                "```",
                "",
            ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
