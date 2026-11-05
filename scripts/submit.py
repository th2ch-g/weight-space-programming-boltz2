import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--partitions", default=os.environ.get("GPU_PARTITIONS"))
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--memory", default="48G")
    args = parser.parse_args()
    if not args.partitions:
        parser.error("Set GPU_PARTITIONS or pass --partitions")
    plan = json.loads(args.plan.read_text())
    if not plan["tasks"]:
        raise ValueError("Empty task plan")
    Path("artifacts").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with Path("artifacts/submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = subprocess.check_output(
            ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%F|%j|%b"],
            text=True,
        )
        parents = set()
        for line in queue.splitlines():
            job, name, resources = line.strip().split("|", 2)
            if name.startswith("wsp-") and "gpu" in resources:
                parents.add(job)
        count = min(4, len({task["id"] for task in plan["tasks"]}))
        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partitions}",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            f"--mem={args.memory}",
            f"--time={args.time}",
            f"--array=0-{count - 1}%4",
            "--job-name=wsp-predict",
            "--output=logs/predict-%A_%a.log",
        ]
        if parents:
            command.append("--dependency=afterany:" + ":".join(sorted(parents)))
        command += ["scripts/slurm.sh", "scripts/worker.py", args.plan.as_posix()]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        receipt = {
            "job_id": job,
            "plan": args.plan.as_posix(),
            "workers": count,
            "dependencies": sorted(parents),
            "max_concurrent_gpu_nodes": 4,
        }
        Path("artifacts", f"submission-{job}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
