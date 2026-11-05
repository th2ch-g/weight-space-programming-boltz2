import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def covariance_components(values):
    means = values.mean(axis=1)
    difference = (means[0] - means[1]) / 2
    even = (means[0] + means[1]) / 2
    variation = (values - means[:, None]).reshape(-1, values.shape[-1])
    parity = difference.T @ difference / len(difference)
    rotation = variation.T @ variation / len(variation)
    return parity, rotation, even.T @ even / len(even)


def direction(parity, rotation):
    ridge = 0.01 * np.trace(rotation) / rotation.shape[0]
    eigenvalues, eigenvectors = eigh(
        parity,
        rotation + ridge * np.eye(rotation.shape[0]),
        subset_by_index=(rotation.shape[0] - 1, rotation.shape[0] - 1),
    )
    vector = eigenvectors[:, 0]
    return vector / np.linalg.norm(vector), float(eigenvalues[0]), float(ridge)


def measure(vector, parity, rotation, even):
    signal = float(vector @ parity @ vector)
    noise = float(vector @ rotation @ vector)
    baseline = float(vector @ even @ vector)
    return {
        "parity_variance": signal,
        "rotation_variance": noise,
        "parity_to_rotation_variance_ratio": signal / max(noise, 1e-30),
        "unprojected_variance_ratio": float(np.trace(parity) / np.trace(rotation)),
        "even_mean_square": baseline,
        "parity_to_all_other_power_ratio": signal / max(noise + baseline, 1e-30),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "summary.json").read_text())
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    groups = {row["id"]: row["group"] for row in manifest["records"]}
    components = {}
    for row in report["systems"]:
        with np.load(args.directory / f"{row['id']}.npz") as arrays:
            components[row["id"]] = covariance_components(arrays["observations"].astype(np.float64))
    full_parity = np.mean([value[0] for value in components.values()], axis=0)
    full_rotation = np.mean([value[1] for value in components.values()], axis=0)
    full_even = np.mean([value[2] for value in components.values()], axis=0)
    full_direction, eigenvalue, ridge = direction(full_parity, full_rotation)
    results = []
    for group in sorted({groups[sid] for sid in components}):
        train = [value for sid, value in components.items() if groups[sid] != group]
        vector, _, _ = direction(
            np.mean([value[0] for value in train], axis=0),
            np.mean([value[1] for value in train], axis=0),
        )
        for sid, (parity, rotation, even) in components.items():
            if groups[sid] == group:
                results.append(
                    {
                        "id": sid,
                        "withheld_group": group,
                        "absolute_cosine_to_full_direction": float(abs(vector @ full_direction)),
                        **measure(vector, parity, rotation, even),
                    }
                )
    output = {
        "scope": "Descriptive symmetry analysis; no weights or predictions are modified.",
        "method": "Generalized eigenvector of parity and rotation covariance, with a fixed one-percent trace ridge. Each system receives equal weight. Validation withholds sequence/scaffold groups.",
        "limitation": "Only the 24 proper cube rotations are sampled. This does not establish invariance under arbitrary rotations or causal use by the denoiser.",
        "full_sample": {
            "regularized_eigenvalue": eigenvalue,
            "ridge": ridge,
            **measure(full_direction, full_parity, full_rotation, full_even),
        },
        "withheld_groups": results,
    }
    (args.directory / "covariance_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    np.save(args.directory / "parity_direction.npy", full_direction)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
