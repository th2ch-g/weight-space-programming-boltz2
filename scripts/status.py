import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("plans", nargs="+", type=Path)
args = parser.parse_args()
for plan_path in args.plans:
    plan = json.loads(plan_path.read_text())
    counts = Counter()
    failures = []
    for task in plan["tasks"]:
        file = Path(task["output"]) / "run.json"
        if file.exists():
            record = json.loads(file.read_text())
            counts[record["status"]] += 1
            if record["status"] == "failed":
                failures.append({"id": task["id"], "error": record.get("error")})
        else:
            counts["pending"] += 1
    print(
        json.dumps(
            {
                "plan": plan_path.as_posix(),
                "planned": len(plan["tasks"]),
                "counts": dict(counts),
                "failures": failures,
            }
        )
    )
