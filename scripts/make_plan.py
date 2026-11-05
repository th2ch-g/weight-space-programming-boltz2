import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["development", "validation", "test"], required=True)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--diagnostic", action="store_true")
parser.add_argument("--trace-layers", action="store_true")
parser.add_argument("--min-protein-length", type=int, default=0)
parser.add_argument("--max-protein-length", type=int, default=100000)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
manifest_path = Path("data/ct209/manifest.json")
manifest = json.loads(manifest_path.read_text())
records = manifest["records"]
if args.pilot:
    if args.split != "development":
        raise ValueError("Pilot predictions are restricted to development")
    by_id = {row["id"]: row for row in records}
    records = [by_id[sid] for sid in manifest["pilot_ids"]]
tasks = []
for row in records:
    if (
        row["split"] != args.split
        or not args.min_protein_length <= row["protein_length"] <= args.max_protein_length
    ):
        continue
    for checkpoint in args.checkpoints:
        path = Path(checkpoint)
        if path.parent != Path(".") or not path.is_file() or path.is_symlink():
            raise ValueError("Use a regular checkpoint in the project root")
        arm = "base" if path.name == "boltz2_base.ckpt" else path.stem.removeprefix("boltz2_wsp_")
        tasks.append(
            {
                "id": row["id"],
                "checkpoint": checkpoint,
                "output": (args.run_root / arm / row["id"]).as_posix(),
                "diagnostic": args.diagnostic,
                "trace_layers": args.trace_layers,
                "seed": 1,
            }
        )
if not tasks:
    raise ValueError("No tasks match the requested split and size range")
plan = {
    "stage": args.split,
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "tasks": tasks,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps({"tasks": len(tasks), "plan": args.output.as_posix()}))
