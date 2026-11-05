import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import Boltz2DiffusionParams, MSAModuleArgs, PairformerArgsV2
from boltz.model.models.boltz2 import Boltz2
from checkpoint_io import load_model_checkpoint
from probe_reference_parity import rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Trunk parity probes require a Slurm GPU allocation")
    torch.set_grad_enabled(False)
    model = load_model_checkpoint(
        Boltz2,
        "boltz2_base.ckpt",
        map_location="cpu",
        strict=True,
        ema=False,
        use_kernels=False,
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(subsample_msa=False, use_paired_feature=True)),
    ).cuda()
    model.eval()
    model.skip_run_structure = True
    model.confidence_prediction = False
    model.affinity_prediction = False
    tasks = [
        task for task in json.loads(args.plan.read_text())["tasks"] if task.get("trace_layers")
    ]
    directory = Path("artifacts/trunk_parity")
    directory.mkdir(exist_ok=True)
    results = []

    def encode(features):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(features, recycling_steps=3)["s"][0].float()

    for task in tasks[: args.limit]:
        root = Path(task["output"])
        processed = root / "prediction/boltz_results_target/processed"
        datamodule = Boltz2InferenceDataModule(
            manifest=Manifest.load(processed / "manifest.json"),
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=Path(".cache/boltz/mols"),
            num_workers=0,
            constraints_dir=processed / "constraints",
            template_dir=processed / "templates",
            extra_mols_dir=processed / "mols",
        )
        features = datamodule.transfer_batch_to_device(
            next(iter(datamodule.predict_dataloader())), torch.device("cuda"), 0
        )
        trace = np.load(root / "layer_trace.npz")
        features["ref_pos"] = torch.from_numpy(trace["ref_pos"]).unsqueeze(0).cuda()
        atom_type = torch.bmm(
            features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        atom_mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
        token_mask = features["mol_type"][0] == 3
        original = encode(features)
        expected = torch.from_numpy(trace["s_trunk"]).cuda()
        error = float((original - expected).norm() / expected.norm())
        if error > 0.01:
            raise ValueError(f"Trunk reconstruction differs from the trace: {error}")
        positions = features["ref_pos"].clone()
        center = positions[0, atom_mask].mean(dim=0, keepdim=True)
        relative = positions[0, atom_mask] - center
        observations = []
        for parity in (1.0, -1.0):
            group = []
            for rotation in rotations():
                features["ref_pos"][0, atom_mask] = parity * relative @ rotation.cuda() + center
                group.append(encode(features)[token_mask].cpu().numpy())
            observations.append(np.stack(group))
        values = np.stack(observations)
        means = values.mean(axis=1)
        difference = means[0] - means[1]
        variation = values - means[:, None]
        gap = float(np.linalg.norm(difference, axis=-1).mean())
        variation_norm = float(np.linalg.norm(variation, axis=-1).mean())
        row = {
            "id": task["id"],
            "rotations_per_parity": 24,
            "trunk_reconstruction_relative_error": error,
            "mean_parity_gap_norm": gap,
            "mean_rotation_variation_norm": variation_norm,
            "parity_gap_to_rotation_variation": gap / max(variation_norm, 1e-12),
        }
        np.savez_compressed(
            directory / f"{task['id']}.npz",
            parity_means=means,
            parity_difference=difference,
            rotation_variance=np.mean(variation**2, axis=(0, 1)),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Development trunk counterfactuals without diffusion or coordinate output.",
                "precision": "GPU bfloat16 mixed, matching prediction",
                "systems": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
