import argparse
import hashlib
import json
from pathlib import Path

import torch
from checkpoint_io import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity-checkpoint", type=Path, default=Path(".cache/boltz/boltz2_aff.ckpt"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    base = load_checkpoint("boltz2_base.ckpt", map_location="cpu", mmap=True)
    affinity = load_checkpoint(args.affinity_checkpoint, map_location="cpu", mmap=True)
    with args.affinity_checkpoint.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    rows = []
    for prefix in ("input_embedder.", "diffusion_conditioning.atom_encoder.", "token_bonds_type."):
        source = {key: value for key, value in base["state_dict"].items() if key.startswith(prefix)}
        same, changed, absent, incompatible = [], [], [], []
        for key, value in source.items():
            other = affinity["state_dict"].get(key)
            if other is None:
                absent.append(key)
            elif value.shape != other.shape:
                incompatible.append(key)
            elif torch.equal(value, other):
                same.append(key)
            else:
                changed.append({
                    "key": key,
                    "relative_difference_norm": float((value.double() - other.double()).norm() / value.double().norm().clamp_min(1e-30)),
                })
        rows.append({
            "prefix": prefix, "identical_tensors": len(same), "changed_tensors": changed,
            "absent_tensors": absent, "incompatible_tensors": incompatible,
        })
    report = {
        "scope": "Read-only comparison of existing pretrained feature encoders; no weights transferred or predictions made.",
        "affinity_checkpoint_sha256": checksum,
        "affinity_hyper_parameters": {
            key: affinity["hyper_parameters"].get(key) for key in (
                "structure_prediction_training", "affinity_prediction", "affinity_ensemble", "skip_run_structure"
            )
        },
        "components": rows,
    }
    Path("artifacts/affinity_encoder_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
