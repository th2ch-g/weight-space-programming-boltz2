import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--base", type=Path, default=Path("boltz2_base.ckpt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    for path in (args.base, args.output):
        if path.resolve().parent != root or path.is_symlink():
            raise ValueError("Checkpoints must be regular files in the project root")
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text())
    before_hash = digest(args.base)
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["state_dict"]
    changed = {}
    total = sum(value.numel() for value in state.values())
    torch.set_grad_enabled(False)
    program_metadata = None
    program_source = None
    if "reserved_channels_program" in spec:
        from program_reserved_channels import reserve_channels

        if "neighbor_bias_program" in spec:
            raise ValueError("Only one analytic program may be applied at a time")
        settings = spec["reserved_channels_program"]
        if (set(settings) not in ({"count"}, {"count", "scope"}, {"count", "scope", "preserve_condition_bias"})
                or settings["count"] != 18):
            raise ValueError("This experiment reserves exactly eighteen channels")
        programmed, program_metadata = reserve_channels(state, **settings)
        program_source = Path(__file__).with_name("program_reserved_channels.py")
        for key, edited in programmed.items():
            original = state[key]
            if edited.shape != original.shape or not torch.isfinite(edited).all():
                raise ValueError("Invalid programmed tensor")
            changed[key] = {
                "changed_elements": int((original != edited).sum()),
                "tensor_elements": original.numel(),
                "delta_norm": float((edited - original).norm()),
                "base_norm": float(original.norm()),
            }
            state[key] = edited
    if "neighbor_bias_program" in spec:
        from program_neighbor_bias import program_neighbor_bias

        settings = spec["neighbor_bias_program"]
        program_source = Path(__file__).with_name("program_neighbor_bias.py")
        if set(settings) != {"strength"} or not 0 <= settings["strength"] <= 2:
            raise ValueError("The neighbor program requires a fixed strength between zero and two")
        programmed = program_neighbor_bias(
            state, prefix="input_embedder.", strength=settings["strength"]
        )
        for key, edited in programmed.items():
            original = state[key]
            if edited.shape != original.shape or not torch.isfinite(edited).all():
                raise ValueError("Invalid programmed tensor")
            changed[key] = {
                "changed_elements": int((original != edited).sum()),
                "tensor_elements": original.numel(),
                "delta_norm": float((edited - original).norm()),
                "base_norm": float(original.norm()),
            }
            state[key] = edited
    for operation in spec["operations"]:
        key = operation["key"]
        if key in changed:
            raise ValueError("Repeated target tensor")
        original = state[key]
        edited = original.clone()
        mask = torch.zeros_like(original, dtype=torch.bool)
        if sum(name in operation for name in ("entries", "columns", "rows")) > 1:
            raise ValueError("Select only one tensor axis")
        if "entries" in operation:
            if original.ndim != 1 or "columns" in operation:
                raise ValueError("Entry ranges require a vector and cannot select columns")
            start, end = operation["entries"]
            if not 0 <= start < end <= original.numel():
                raise ValueError("Invalid vector entry range")
            mask[start:end] = True
        elif "columns" in operation:
            start, end = operation["columns"]
            if original.ndim != 2 or not 0 <= start < end <= original.shape[1]:
                raise ValueError("Invalid matrix column range")
            mask[:, start:end] = True
        elif "rows" in operation:
            start, end = operation["rows"]
            if original.ndim != 2 or not 0 <= start < end <= original.shape[0]:
                raise ValueError("Invalid matrix row range")
            mask[start:end] = True
        else:
            mask.fill_(True)
        if "contrast_row" in operation:
            if "rows" not in operation:
                raise ValueError("Row contrast requires an explicit row range")
            reference = operation["contrast_row"]
            if not 0 <= reference < original.shape[0] or start <= reference < end:
                raise ValueError("The reference row must exist and remain outside the edit")
            edited[start:end] = original[reference] + operation["factor"] * (
                original[start:end] - original[reference]
            )
        elif "remove_projection" in operation:
            projection = operation["remove_projection"]
            first, last = projection["basis_columns"]
            basis = state[projection["basis_key"]][:, first:last].double()
            q, _ = torch.linalg.qr(basis, mode="reduced")
            first, last = operation["columns"]
            columns = original[:, first:last].double()
            edited[:, first:last] -= (q @ (q.T @ columns)).to(original.dtype)
        elif "gain_orthogonal_to" in operation:
            first, last = operation["columns"]
            vectors = []
            for source in operation["gain_orthogonal_to"]:
                begin, end = source["entries"]
                vector = state[source["key"]][begin:end].double()
                if vector.ndim != 1 or vector.numel() != last - first:
                    raise ValueError("Preserved directions must match the selected columns")
                vectors.append(vector)
            basis = torch.stack(vectors, dim=1)
            q, r = torch.linalg.qr(basis, mode="reduced")
            if torch.min(torch.abs(r.diag())) < 1e-8:
                raise ValueError("Preserved directions are linearly dependent")
            columns = original[:, first:last].double()
            residual = columns - (columns @ q) @ q.T
            edited[:, first:last] += ((operation["factor"] - 1) * residual).to(original.dtype)
            delta = edited[:, first:last].double() - columns
            if (delta @ q).abs().max() > 1e-4:
                raise ValueError("Edited operator changed a preserved direction")
        else:
            edited[mask] *= operation["factor"]
        if not torch.isfinite(edited).all():
            raise ValueError("Nonfinite edited tensor")
        assert torch.equal(original[~mask], edited[~mask])
        changed[key] = {
            "changed_elements": int((original != edited).sum()),
            "tensor_elements": original.numel(),
            "delta_norm": float((edited - original).norm()),
            "base_norm": float(original.norm()),
        }
        state[key] = edited
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    saved = torch.load(args.output, map_location="cpu", weights_only=False, mmap=True)
    original = torch.load(args.base, map_location="cpu", weights_only=False, mmap=True)
    assert saved.keys() == original.keys()
    assert saved["state_dict"].keys() == original["state_dict"].keys()
    for key in state:
        assert torch.equal(saved["state_dict"][key], state[key]), key
        if key not in changed:
            assert torch.equal(saved["state_dict"][key], original["state_dict"][key]), key
    assert digest(args.base) == before_hash
    count = sum(v["changed_elements"] for v in changed.values())
    record = {
        "spec": spec,
        "editor_sha256": digest(Path(__file__)),
        "program_source_sha256": digest(program_source) if program_source else None,
        "program_metadata": program_metadata,
        "base_sha256": before_hash,
        "output_sha256": digest(args.output),
        "output": args.output.name,
        "changed": changed,
        "changed_elements": count,
        "total_elements": total,
        "changed_fraction": count / total,
        "base_unchanged": True,
        "all_untargeted_tensors_equal": True,
        "status": "candidate_unvalidated",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts", args.output.stem + ".edit.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in record.items() if k not in {"spec", "changed"}}))


if __name__ == "__main__":
    main()
