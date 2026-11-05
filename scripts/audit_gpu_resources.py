import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True)
    args = parser.parse_args()
    names = ("wsp-smoke", "wsp-predict", "wsp-trunk-parity")
    output = subprocess.check_output(
        [
            "sacct", "-X", "--starttime", args.since, "--user", os.environ["USER"],
            "--name", ",".join(names), "--noheader", "--parsable2",
            "--format=JobID,JobName,State,Start,End,AllocNodes,NodeList",
        ],
        text=True,
    )
    rows, events, pending = [], [], []
    now = datetime.now()
    for line in output.splitlines():
        job, name, state, start, end, nodes, node_list = line.split("|")
        if name not in names:
            raise ValueError("Unexpected job in scheduler audit")
        row = dict(zip(
            ("job_id", "name", "state", "start", "end", "nodes", "node_list"),
            (job, name, state, start, end, nodes, node_list), strict=True,
        ))
        rows.append(row)
        if start in ("Unknown", "None", ""):
            pending.append(job)
            continue
        begin = datetime.fromisoformat(start)
        finish = datetime.fromisoformat(end) if end not in ("Unknown", "None", "") else now
        if finish > begin and int(nodes):
            events.extend(((begin, int(nodes), job), (finish, -int(nodes), job)))
    count, maximum = 0, 0
    active = set()
    peak_jobs = []
    for _, delta, job in sorted(events):
        count += delta
        if delta > 0:
            active.add(job)
        else:
            active.remove(job)
        if count > maximum:
            maximum = count
            peak_jobs = sorted(active)
    report = {
        "scope": "Allocated nodes for the project's known GPU job names; accounting does not expose GPU TRES on this cluster.",
        "scheduler_job_names": names,
        "since": args.since,
        "maximum_simultaneous_allocated_nodes": maximum,
        "limit": 4,
        "within_limit": maximum <= 4,
        "peak_jobs": peak_jobs,
        "all_jobs_ended": not pending and all(row["end"] not in ("Unknown", "None", "") for row in rows),
        "pending_jobs": pending,
        "rows": rows,
    }
    Path("artifacts/gpu_resource_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    if not report["within_limit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
