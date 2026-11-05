import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--destination", type=Path, default=Path(".cache/boltz"))
args = parser.parse_args()
source = args.source.expanduser().resolve()
destination = args.destination.resolve()
root = Path.cwd().resolve()
if not destination.is_relative_to(root):
    raise ValueError("Cache links must be inside the project")
base = root / "boltz2_base.ckpt"
if not base.is_file() or base.is_symlink():
    raise ValueError("Prepare a regular local base checkpoint first")
resources = {name: source / name for name in ("mols", "mols.tar", "ccd.pkl", "boltz2_aff.ckpt")}
resources["boltz2_conf.ckpt"] = base
for name, target in resources.items():
    if not target.exists():
        raise FileNotFoundError(target)
    link = destination / name
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise FileExistsError(link)
destination.mkdir(parents=True, exist_ok=True)
for name, target in resources.items():
    link = destination / name
    if not link.exists():
        link.symlink_to(os.path.relpath(target, destination))
print("Local cache links verified; external resources were not modified")
