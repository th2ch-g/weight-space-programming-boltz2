import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = ("atom_encoder_token_output", "single_conditioning_addition", "token_layer_0_input",
              "token_layer_12_input", "final_token_norm_input")
    rows = []
    for task in json.loads(args.plan.read_text())["tasks"]:
        root = Path(task["output"])
        run = json.loads((root / "run.json").read_text())
        if run["status"] != "complete" or not run["diagnostic"] or run["reference_reflection"]:
            raise ValueError("Expected completed, unreflected diagnostic predictions")
        trace_path = root / "reservation_trace.json"
        trace = json.loads(trace_path.read_text())
        if trace["diffusion_calls_observed"] != 200:
            raise ValueError("Unexpected diffusion call count")
        for group in ("protein", "ligand"):
            for label in labels:
                selected = [row for row in trace["rows"] if row["group"] == group and row["label"] == label]
                if {row["step"] for row in selected} != {0, 50, 100, 150, 199}:
                    raise ValueError("Missing diagnostic time points")
                largest = max(selected, key=lambda row: max(row["selected_rms"]))
                index = max(range(len(trace["token_slots"])), key=lambda i: largest["selected_rms"][i])
                rows.append({
                    "id": task["id"], "checkpoint": task["checkpoint"], "group": group, "label": label,
                    "median_selected_centered_variance_fraction": statistics.median(row["mean_centered_variance_fraction_in_selected"] for row in selected),
                    "median_std_ratio_if_selected_zeroed": statistics.median(row["mean_std_ratio_if_selected_zeroed"] for row in selected),
                    "largest_rms_slot": trace["token_slots"][index], "largest_rms_step": largest["step"],
                    "largest_slot_mean": largest["selected_mean"][index], "largest_slot_std": largest["selected_std"][index],
                    "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    "prediction_sha256": run["prediction_sha256"],
                    "runner_sha256": run["runner_sha256"],
                })
    report = {
        "scope": "Read-only activation diagnostic, not chemistry evaluation or a repair result.",
        "interpretation": "A low final readout norm does not imply that an earlier residual channel is negligible. Zeroing reader columns after AdaLN also removes learned condition-bias contributions; that algebraic effect is separate from the measured normalization change.",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"predictions": len({(row["id"], row["checkpoint"]) for row in rows}), "summary_rows": len(rows)}))


if __name__ == "__main__":
    main()
