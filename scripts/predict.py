import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_with_local_molecule(source, destination, molecule):
    from boltz.data import const

    (destination / "mols").mkdir(parents=True)
    for name in ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar", "ccd.pkl"):
        target = destination / name
        target.symlink_to(os.path.relpath(source / name, target.parent))
    for name in const.canonical_tokens:
        target = destination / "mols" / f"{name}.pkl"
        target.symlink_to(os.path.relpath(source / "mols" / target.name, target.parent))
    target = destination / "mols/LIG1.pkl"
    target.symlink_to(os.path.relpath(molecule, target.parent))
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--read-checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--reference-reflection", action="store_true")
    parser.add_argument("--reflection-path", choices=("all", "input", "denoiser"), default="all")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--trace-reservation", type=Path)
    args = parser.parse_args()
    if args.trace_reservation and not args.diagnostic:
        parser.error("Reservation profiling is restricted to diagnostic runs")
    reservation_selection = (
        json.loads(args.trace_reservation.read_text())["program_metadata"]
        if args.trace_reservation else None
    )
    if args.reference_reflection and not args.diagnostic:
        parser.error("Reference reflection is restricted to diagnostic runs")
    if args.reflection_path != "all" and not args.reference_reflection:
        parser.error("A reflection path requires an explicit diagnostic reference reflection")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU predictions require a Slurm GPU allocation")
    if version("boltz") != "2.2.1":
        raise RuntimeError("Expected boltz==2.2.1")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(Path("data/ct209/manifest.json").read_text())
    row = next(x for x in manifest["records"] if x["id"] == args.id)
    args.output.mkdir(parents=True)
    doc = yaml.safe_load(Path(row["input"]).read_text())
    protein = next(x["protein"] for x in doc["sequences"] if "protein" in x)
    protein["msa"] = os.path.relpath(Path("data/ct209/msa") / f"{args.id}.a3m", args.output)
    checkpoint = (args.read_checkpoint or args.checkpoint).resolve()
    checkpoint_sha256 = digest(checkpoint)
    if args.read_checkpoint and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("Node-local checkpoint hash mismatch")
    cache = args.cache.resolve()
    output = args.output.resolve()
    local_molecule = Path("data/ct209/local_mols") / f"{args.id}.pkl"
    if local_molecule.exists():
        ligand = next(x["ligand"] for x in doc["sequences"] if "ligand" in x)
        ligand.pop("smiles")
        ligand["ccd"] = "LIG1"
        cache = cache_with_local_molecule(cache, output / "cache", local_molecule.resolve())
    (output / "target.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    from boltz import main as boltz_main
    from boltz.model.models.boltz2 import Boltz2

    metadata = {
        "id": args.id,
        "split": row["split"],
        "seed": args.seed,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "node_local_checkpoint_cache": args.read_checkpoint is not None,
        "input_sha256": row["input_sha256"],
        "msa_sha256": row["msa_sha256"],
        "staged_input_sha256": digest(output / "target.yaml"),
        "local_molecule_sha256": digest(local_molecule) if local_molecule.exists() else None,
        "input_name_overflow_fallback": local_molecule.exists(),
        "runner_sha256": digest(Path(__file__)),
        "settings": {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_msa_seqs": 8192,
            "subsample_msa": False,
            "use_kernels": False,
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": False,
        },
        "versions": {k: version(k) for k in ("boltz", "torch", "rdkit")},
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "source_sha256": digest(inspect.getfile(Boltz2)),
        "job_id": os.environ["SLURM_JOB_ID"],
        "diagnostic": args.diagnostic,
        "reference_reflection": args.reference_reflection,
        "reflection_path": args.reflection_path if args.reference_reflection else "none",
        "trace_layers": args.trace_layers,
        "reservation_selection_sha256": digest(args.trace_reservation) if args.trace_reservation else None,
        "reservation_trace_source_sha256": digest(Path(__file__).with_name("trace_reservation.py"))
        if args.trace_reservation else None,
        "status": "running",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    handles = []
    diag = {}
    frames = []
    diagnostic_state = {"step": 0}
    layer_traces = []
    reservation_traces = []

    class Loader:
        @staticmethod
        def load_from_checkpoint(*pos, **kwargs):
            for key in ("fk_steering", "physical_guidance_update", "contact_guidance_update"):
                kwargs["steering_args"][key] = False
            model = Boltz2.load_from_checkpoint(*pos, **kwargs)
            if args.reference_reflection:

                def reflect_reference(module, inputs, named):
                    if diagnostic_state.get("reference_reflected"):
                        return
                    feats = named.get("feats", inputs[0] if inputs else None)
                    atom_type = torch.bmm(
                        feats["atom_to_token"].float(), feats["mol_type"].float().unsqueeze(-1)
                    ).squeeze(-1)
                    mask = (atom_type[0] == 3) & feats["atom_pad_mask"][0].bool()
                    positions = feats["ref_pos"].clone()
                    center = positions[0, mask, 0].mean()
                    positions[0, mask, 0] = 2 * center - positions[0, mask, 0]
                    diagnostic_state["original_reference"] = feats["ref_pos"]
                    diagnostic_state["reference_features"] = feats
                    feats["ref_pos"] = positions
                    diagnostic_state["reference_reflected"] = True

                reflection_target = (
                    model.diffusion_conditioning.atom_encoder
                    if args.reflection_path == "denoiser" else model.input_embedder
                )
                handles.append(
                    reflection_target.register_forward_pre_hook(
                        reflect_reference, with_kwargs=True
                    )
                )
            if args.trace_layers:
                from trace_layers import LayerTrace

                layer_traces.append(LayerTrace(model))
            if reservation_selection:
                from trace_reservation import ReservationTrace

                reservation_traces.append(ReservationTrace(model, reservation_selection))
            if args.diagnostic:

                def observe_input_reference(module, inputs, named):
                    if "input_ref_pos" in diag:
                        return
                    feats = named.get("feats", inputs[0] if inputs else None)
                    atom_type = torch.bmm(
                        feats["atom_to_token"].float(), feats["mol_type"].float().unsqueeze(-1)
                    ).squeeze(-1)
                    mask = (atom_type[0] == 3) & feats["atom_pad_mask"][0].bool()
                    diag["input_ref_pos"] = feats["ref_pos"][0, mask].float().cpu().numpy()

                def observe_reference(module, inputs, named, result):
                    feats = named["feats"]
                    atom_type = torch.bmm(
                        feats["atom_to_token"].float(), feats["mol_type"].float().unsqueeze(-1)
                    ).squeeze(-1)
                    mask = (atom_type[0] == 3) & feats["atom_pad_mask"][0].bool()
                    diagnostic_state["mask"] = mask
                    diag["ref_pos"] = feats["ref_pos"][0, mask].float().cpu().numpy()
                    diag["denoiser_ref_pos"] = diag["ref_pos"].copy()
                    diag["ref_element"] = feats["ref_element"][0, mask].argmax(-1).cpu().numpy()
                    q, c, _p, _ = result
                    ref_q = torch.nn.functional.linear(
                        feats["ref_pos"].float(), module.embed_atom_features.weight[:, :3]
                    )
                    diag["component_norms"] = np.array(
                        [
                            ref_q[0, mask].norm(dim=-1).mean().item(),
                            q[0, mask].norm(dim=-1).mean().item(),
                            c[0, mask].norm(dim=-1).mean().item(),
                        ]
                    )

                def observe_score(module, inputs, named, result):
                    step = diagnostic_state["step"]
                    diagnostic_state["step"] += 1
                    if step not in {0, 25, 50, 75, 100, 125, 150, 175, 199}:
                        return
                    sigma = (named["times"].float() * 4).exp() * model.structure_module.sigma_data
                    sigma = sigma.reshape(-1, 1, 1)
                    x = named["r_noisy"] / model.structure_module.c_in(sigma)
                    x0 = (
                        model.structure_module.c_skip(sigma) * x
                        + model.structure_module.c_out(sigma) * result
                    )
                    frames.append(
                        (
                            step,
                            float(sigma[0, 0, 0]),
                            x0[0, diagnostic_state["mask"]].float().cpu().numpy(),
                        )
                    )

                handles.append(
                    model.input_embedder.register_forward_pre_hook(
                        observe_input_reference, with_kwargs=True
                    )
                )
                handles.append(
                    model.diffusion_conditioning.atom_encoder.register_forward_hook(
                        observe_reference, with_kwargs=True
                    )
                )
                handles.append(
                    model.structure_module.score_model.register_forward_hook(
                        observe_score, with_kwargs=True
                    )
                )
                if args.reference_reflection and args.reflection_path != "all":

                    def restore_reference(module, inputs, named, result):
                        feats = diagnostic_state["reference_features"]
                        feats["ref_pos"] = diagnostic_state["original_reference"]
                        diagnostic_state["reference_restored"] = True

                    handles.append(
                        reflection_target.register_forward_hook(restore_reference, with_kwargs=True)
                    )
            return model

    started = time.monotonic()
    previous = Path.cwd()
    try:
        os.chdir(output)
        with patch.object(boltz_main, "Boltz2", Loader):
            boltz_main.predict.callback(
                data="target.yaml",
                out_dir="prediction",
                cache=str(cache),
                checkpoint=str(checkpoint),
                model="boltz2",
                devices=1,
                accelerator="gpu",
                num_workers=0,
                recycling_steps=3,
                sampling_steps=200,
                diffusion_samples=1,
                max_parallel_samples=1,
                seed=args.seed,
                use_msa_server=False,
                no_kernels=True,
                subsample_msa=False,
                max_msa_seqs=8192,
                preprocessing_threads=1,
                use_potentials=False,
                output_format="pdb",
            )
        pred = output / "prediction/boltz_results_target/predictions/target/target_model_0.pdb"
        if not pred.is_file():
            raise RuntimeError("Prediction missing despite predictor exit")
        if args.reference_reflection and not diagnostic_state.get("reference_reflected"):
            raise RuntimeError("Diagnostic reference reflection was not applied")
        if (
            args.reference_reflection and args.reflection_path != "all"
            and not diagnostic_state.get("reference_restored")
        ):
            raise RuntimeError("Diagnostic reference coordinates were not restored")
        metadata.update(
            status="complete",
            prediction=pred.relative_to(output).as_posix(),
            prediction_sha256=digest(pred),
        )
    except BaseException as exc:
        metadata.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc).replace(str(previous), "<project>"),
        )
        raise
    finally:
        os.chdir(previous)
        for handle in handles:
            handle.remove()
        for trace in layer_traces:
            trace.save(output / "layer_trace.npz")
        for trace in reservation_traces:
            trace.save(output / "reservation_trace.json")
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metadata["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.diagnostic and diag:
            if frames:
                diag.update(
                    steps=np.array([x[0] for x in frames]),
                    sigmas=np.array([x[1] for x in frames]),
                    x0=np.stack([x[2] for x in frames]),
                )
            np.savez_compressed(output / "diagnostics.npz", **diag)


if __name__ == "__main__":
    main()
