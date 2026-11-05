import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan_text = args.plan.read_text()
    plan = json.loads(plan_text)
    plan_hash = hashlib.sha256(plan_text.encode()).hexdigest()
    assert int(os.environ["SLURM_ARRAY_TASK_COUNT"]) <= 4
    ids = list(dict.fromkeys(t["id"] for t in plan["tasks"]))
    claims_file = Path("artifacts") / f"claims-{os.environ['SLURM_ARRAY_JOB_ID']}.json"
    failures = 0
    with tempfile.TemporaryDirectory(prefix="wsp-read-cache-") as temporary:
        sources = {t["checkpoint"]: Path(t["checkpoint"]) for t in plan["tasks"]}
        required = sum(p.stat().st_size for p in sources.values())
        copies = {}
        if shutil.disk_usage(temporary).free > required + 2 * 1024**3:
            for name, source in sources.items():
                destination = Path(temporary) / source.name
                source_hash = digest(source)
                shutil.copyfile(source, destination)
                if digest(destination) != source_hash:
                    raise ValueError("Checkpoint cache copy differs from project checkpoint")
                destination.chmod(0o444)
                copies[name] = (destination, source_hash)
        while True:
            with claims_file.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                previous = handle.read()
                claims = json.loads(previous) if previous else {"next": 0, "plan_sha256": plan_hash}
                assert claims["plan_sha256"] == plan_hash
                index = claims["next"]
                if index >= len(ids):
                    break
                claims["next"] += 1
                handle.seek(0)
                handle.truncate()
                json.dump(claims, handle)
                handle.flush()
            for task in [t for t in plan["tasks"] if t["id"] == ids[index]]:
                command = [
                    sys.executable,
                    "scripts/predict.py",
                    "--id",
                    task["id"],
                    "--checkpoint",
                    task["checkpoint"],
                    "--output",
                    task["output"],
                    "--seed",
                    str(task.get("seed", 1)),
                ]
                if task.get("diagnostic"):
                    command.append("--diagnostic")
                if task.get("reference_reflection"):
                    command += [
                        "--reference-reflection", "--reflection-path",
                        task.get("reflection_path", "all"),
                    ]
                if task.get("trace_layers"):
                    command.append("--trace-layers")
                if task.get("trace_reservation"):
                    command += ["--trace-reservation", task["trace_reservation"]]
                if task["checkpoint"] in copies:
                    copy, checksum = copies[task["checkpoint"]]
                    command += [
                        "--read-checkpoint",
                        str(copy),
                        "--expected-checkpoint-sha256",
                        checksum,
                    ]
                code = subprocess.run(command, check=False).returncode
                failures += code != 0
                if code:
                    record_path = Path(task["output"]) / "run.json"
                    record = json.loads(record_path.read_text()) if record_path.exists() else {}
                    if record.get("status") != "failed":
                        record.update(id=task["id"], status="failed", process_returncode=code)
                        record_path.parent.mkdir(parents=True, exist_ok=True)
                        record_path.write_text(json.dumps(record, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
