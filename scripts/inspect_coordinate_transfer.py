import argparse
import json
from pathlib import Path

import torch


def align(source, target):
    left, singular, right = torch.linalg.svd(target @ source.T, full_matrices=False)
    return left @ right, singular


def compare(source, target, mapping):
    projected = mapping @ source
    scale = (projected * target).sum() / projected.square().sum().clamp_min(1e-30)
    residual = (scale * projected - target).norm() / target.norm().clamp_min(1e-30)
    cosine = (projected * target).sum() / (
        projected.norm() * target.norm()
    ).clamp_min(1e-30)
    return {
        "cosine": float(cosine),
        "relative_residual_after_scalar_alignment": float(residual),
        "scalar_alignment": float(scale),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("boltz2_base.ckpt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/coordinate_transfer_inspection.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["state_dict"]
    source = state["input_embedder.atom_encoder.embed_atom_features.weight"].double()
    target = state["diffusion_conditioning.atom_encoder.embed_atom_features.weight"].double()
    noisy = state["structure_module.score_model.atom_attention_encoder.r_to_q_trans.weight"].double()
    if source.shape != target.shape or source.shape[1] != 388:
        raise ValueError("Expected matching atom encoders with 388 feature columns")
    q, _ = torch.linalg.qr(noisy, mode="reduced")
    groups = {
        "charge_and_elements": list(range(3, 132)),
        "atom_names": list(range(132, 388)),
    }
    rows = []
    for name, columns in groups.items():
        complement = [column for column in range(3, 388) if column not in columns]
        mapping, singular = align(source[:, columns], target[:, columns])
        geometry = mapping @ source[:, :3]
        rows.append({
            "alignment_columns": name,
            "alignment": compare(source[:, columns], target[:, columns], mapping),
            "held_out_chemical_columns": compare(source[:, complement], target[:, complement], mapping),
            "mapped_geometry_norm": float(geometry.norm()),
            "geometry_in_noisy_coordinate_subspace_fraction": float((q @ (q.T @ geometry)).norm().square() / geometry.norm().square()),
            "singular_values": singular.tolist(),
        })
    mapping, singular = align(source[:, 3:], target[:, 3:])
    mapped = mapping @ source[:, :3]
    report = {
        "scope": "Read-only weight-space basis-transfer diagnostic. No activation, label, optimizer, checkpoint edit, or inference is used.",
        "method": "Use the orthogonal polar factor of the cross-product between corresponding chemical feature columns. Evaluate transfer separately on columns excluded from that factor.",
        "limitation": "Chemical basis agreement would not prove geometric equivariance or downstream use of handedness.",
        "source_coordinate_norm": float(source[:, :3].norm()),
        "target_coordinate_norm": float(target[:, :3].norm()),
        "noisy_coordinate_norm": float(noisy.norm()),
        "source_chemical_norm": float(source[:, 3:].norm()),
        "target_chemical_norm": float(target[:, 3:].norm()),
        "all_chemical_alignment": compare(source[:, 3:], target[:, 3:], mapping),
        "mapped_geometry_in_noisy_coordinate_subspace_fraction": float((q @ (q.T @ mapped)).norm().square() / mapped.norm().square()),
        "original_geometry_in_noisy_coordinate_subspace_fraction": float((q @ (q.T @ target[:, :3])).norm().square() / target[:, :3].norm().square()),
        "groups": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}), flush=True)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != "singular_values"}), flush=True)


if __name__ == "__main__":
    main()
