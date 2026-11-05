import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.model.modules.trunkv2 import InputEmbedder


def rotations():
    matrices = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.eye(3)[list(order)] * np.asarray(signs)[:, None]
            if np.linalg.det(matrix) > 0:
                matrices.append(torch.tensor(matrix, dtype=torch.float32))
    assert len(matrices) == 24
    return matrices


def load_encoder():
    checkpoint = torch.load("boltz2_base.ckpt", map_location="cpu", weights_only=False, mmap=True)
    parameters = checkpoint["hyper_parameters"]
    keys = (
        "atom_s",
        "atom_z",
        "token_s",
        "token_z",
        "atoms_per_window_queries",
        "atoms_per_window_keys",
        "atom_feature_dim",
    )
    model = InputEmbedder(**{key: parameters[key] for key in keys}, **parameters["embedder_args"])
    model.load_state_dict(
        {
            key.removeprefix("input_embedder."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("input_embedder.")
        }
    )
    model.eval()
    return model


def load_features(root):
    processed = root / "prediction/boltz_results_target/processed"
    metadata = json.loads((root / "run.json").read_text())
    molecule_digest = metadata.get("local_molecule_sha256")
    mol_dir = root / "cache/mols" if molecule_digest else Path(".cache/boltz/mols")
    if molecule_digest:
        with (mol_dir / "LIG1.pkl").open("rb") as handle:
            observed = hashlib.file_digest(handle, "sha256").hexdigest()
        if observed != molecule_digest:
            raise ValueError("Diagnostic local molecule differs from the prediction input")
    datamodule = Boltz2InferenceDataModule(
        manifest=Manifest.load(processed / "manifest.json"),
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
    )
    features = next(iter(datamodule.predict_dataloader()))
    with np.load(root / "layer_trace.npz") as trace:
        features["ref_pos"] = torch.from_numpy(trace["ref_pos"]).unsqueeze(0)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--neurons", action="store_true")
    parser.add_argument("--save-observations", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference_parity"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = load_encoder()
    captures = {}
    handles = []
    layers = model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers

    def capture(index):
        def observe(module, inputs):
            value = inputs[0]
            captures[index] = value.reshape(-1, value.shape[-1]).numpy().copy()

        return observe

    if args.neurons:
        for index, layer in enumerate(layers):
            handles.append(layer.transition.b_to_a.register_forward_pre_hook(capture(index)))
    tasks = [
        task for task in json.loads(args.plan.read_text())["tasks"] if task.get("trace_layers")
    ]
    results = []
    output = args.output
    output.mkdir(exist_ok=True)
    for task in tasks[: args.limit]:
        root = Path(task["output"])
        features = load_features(root)
        trace = np.load(root / "layer_trace.npz")
        atom_type = torch.bmm(
            features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        atom_mask = (atom_type[0] == 3) & features["atom_pad_mask"][0].bool()
        token_mask = features["mol_type"][0] == 3
        original = model(features)[0]
        protein_mask = ((atom_type[0] == 0) & features["atom_pad_mask"][0].bool()).numpy()
        protein_neurons = (
            np.stack([np.mean(captures[i][protein_mask] ** 2, axis=0) for i in range(len(layers))])
            if args.neurons
            else None
        )
        expected = torch.from_numpy(trace["s_inputs"])
        reference_error = float((original - expected).norm() / expected.norm())
        if reference_error > 0.05:
            raise ValueError(f"Input reconstruction differs from the trace: {reference_error}")
        positions = features["ref_pos"].clone()
        center = positions[0, atom_mask].mean(dim=0, keepdim=True)
        relative = positions[0, atom_mask] - center
        observations = []
        neuron_observations = []
        for parity in (1.0, -1.0):
            group = []
            neuron_group = []
            for rotation in rotations():
                features["ref_pos"][0, atom_mask] = parity * relative @ rotation + center
                group.append(model(features)[0, token_mask].numpy())
                if args.neurons:
                    neuron_group.append(
                        np.stack([captures[i][atom_mask.numpy()] for i in range(len(layers))])
                    )
            observations.append(np.stack(group))
            if args.neurons:
                neuron_observations.append(np.stack(neuron_group))
        values = np.stack(observations)
        means = values.mean(axis=1)
        difference = means[0] - means[1]
        variation = values - means[:, None]
        gap = float(np.linalg.norm(difference, axis=-1).mean())
        variation_norm = float(np.linalg.norm(variation, axis=-1).mean())
        row = {
            "id": task["id"],
            "rotations_per_parity": 24,
            "input_reconstruction_relative_error": reference_error,
            "mean_parity_gap_norm": gap,
            "mean_rotation_variation_norm": variation_norm,
            "parity_gap_to_rotation_variation": gap / max(variation_norm, 1e-12),
        }
        extra = {}
        if args.save_observations:
            extra["observations"] = values
        if args.neurons:
            neuron_values = np.stack(neuron_observations)
            neuron_means = neuron_values.mean(axis=1)
            extra = {
                "neuron_parity_difference": neuron_means[0] - neuron_means[1],
                "neuron_rotation_variance": np.mean(
                    (neuron_values - neuron_means[:, None]) ** 2, axis=(0, 1)
                ),
                "protein_neuron_mean_square": protein_neurons,
                "neuron_output_column_norm": np.stack(
                    [layer.transition.b_to_a.weight.norm(dim=0).numpy() for layer in layers]
                ),
            }
        np.savez_compressed(
            output / f"{task['id']}.npz",
            parity_means=means,
            parity_difference=difference,
            rotation_variance=np.mean(variation**2, axis=(0, 1)),
            **extra,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "scope": "Development input-encoder counterfactuals; reflections are diagnostic only.",
        "precision": "CPU float32",
        "interpretation": "A parity gap does not by itself establish that downstream layers use it.",
        "systems": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
