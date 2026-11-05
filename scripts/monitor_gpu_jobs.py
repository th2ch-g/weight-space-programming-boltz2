import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def expand_watched_jobs(jobs, receipt_directory=Path("artifacts")):
    expanded = []
    for job in jobs:
        receipt_path = receipt_directory / f"submission-{job}.json"
        if not receipt_path.exists():
            expanded.append(job)
            continue
        receipt = json.loads(receipt_path.read_text())
        count = receipt["workers"]
        if str(receipt["job_id"]) != job or not isinstance(count, int) or not 1 <= count <= 4:
            raise ValueError("Invalid GPU array submission receipt")
        expanded.extend(f"{job}_{index}" for index in range(count))
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    if args.seconds <= 0 or not 1 <= args.interval <= 60:
        parser.error("Use a positive duration and an interval between one and sixty seconds")
    started = time.monotonic()
    watched = expand_watched_jobs(args.jobs)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    observations = []
    while time.monotonic() - started < args.seconds:
        queue = subprocess.check_output(
            ["squeue", "--user", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%D|%b"], text=True,
        )
        active = []
        for line in queue.splitlines():
            job, name, state, nodes, resources = line.strip().split("|")
            if name.startswith("wsp-") and "gpu" in resources and state in {"RUNNING", "COMPLETING"}:
                active.append({"job_id": job, "name": name, "state": state, "nodes": int(nodes)})
        accounting = subprocess.check_output(
            ["sacct", "-X", "--jobs", ",".join(args.jobs), "--noheader", "--parsable2", "--format=JobID,State,ExitCode"], text=True,
        )
        statuses = {}
        for line in accounting.splitlines():
            job, state, exit_code = line.split("|")
            statuses[job] = {"state": state.split()[0], "exit_code": exit_code}
        nodes = sum(row["nodes"] for row in active)
        observations.append({"time_utc": datetime.now(UTC).isoformat(), "allocated_nodes": nodes, "active_gpu_jobs": active, "watched_jobs": statuses})
        done = all(statuses.get(job, {}).get("state") in terminal for job in watched)
        report = {"limit": 4, "requested_jobs": args.jobs, "watched_jobs": watched, "all_watched_jobs_terminal": done, "maximum_observed_nodes": max(row["allocated_nodes"] for row in observations), "observations": observations}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if nodes > 4:
            raise RuntimeError(f"Project GPU node limit exceeded: {nodes}")
        if done:
            print(json.dumps({key: value for key, value in report.items() if key != "observations"}))
            return
        time.sleep(args.interval)
    raise TimeoutError("The bounded monitor expired before all watched jobs became terminal")


if __name__ == "__main__":
    main()
