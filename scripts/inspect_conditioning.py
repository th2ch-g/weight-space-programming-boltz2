import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    state = torch.load("boltz2_base.ckpt", map_location="cpu", weights_only=False, mmap=True)[
        "state_dict"
    ]
    prefix = "structure_module.score_model.single_conditioner."
    gamma = state[prefix + "norm_single.weight"]
    beta = state[prefix + "norm_single.bias"]
    weight = state[prefix + "single_embed.weight"]
    summary = {
        "trunk_norm_gamma": float(gamma[:384].norm()),
        "input_norm_gamma": float(gamma[384:].norm()),
        "trunk_projection_norm": float(weight[:, :384].norm()),
        "input_projection_norm": float(weight[:, 384:].norm()),
        "systems": [],
    }
    for task in json.loads(args.plan.read_text())["tasks"]:
        if not task.get("trace_layers"):
            continue
        data = np.load(Path(task["output"]) / "layer_trace.npz")
        s = torch.from_numpy(np.concatenate([data["s_trunk"], data["s_inputs"]], axis=-1))
        z = F.layer_norm(s, (768,), gamma, beta)
        centered = F.layer_norm(s, (768,)) * gamma
        mean = s.mean(dim=-1, keepdim=True)
        scale = torch.sqrt(s.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
        contributions = {
            "trunk": F.linear(z[:, :384], weight[:, :384]),
            "input": F.linear(z[:, 384:], weight[:, 384:]),
            "trunk_without_beta": F.linear(centered[:, :384], weight[:, :384]),
            "input_without_beta": F.linear(centered[:, 384:], weight[:, 384:]),
            "input_signal": F.linear(s[:, 384:] * gamma[384:] / scale, weight[:, 384:]),
            "input_centering": F.linear(-mean * gamma[384:] / scale, weight[:, 384:]),
        }
        row = {"id": task["id"]}
        for label, mol_type in (("protein", 0), ("ligand", 3)):
            mask = torch.from_numpy(data["mol_type"] == mol_type)
            for name, value in contributions.items():
                row[f"{label}_{name}_norm"] = float(value[mask].norm(dim=-1).mean())
        summary["systems"].append(row)
    Path("artifacts/conditioning_inspection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
