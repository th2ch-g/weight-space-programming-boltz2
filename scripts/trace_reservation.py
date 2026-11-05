import json

import torch


def channel_statistics(values, slots):
    values = values.float()
    selected = values[:, slots]
    total_sum = values.sum(-1)
    total_square = values.square().sum(-1)
    mean = total_sum / values.shape[-1]
    variance = (total_square / values.shape[-1] - mean.square()).clamp_min(0)
    zeroed_mean = (total_sum - selected.sum(-1)) / values.shape[-1]
    zeroed_variance = ((total_square - selected.square().sum(-1)) / values.shape[-1] - zeroed_mean.square()).clamp_min(0)
    centered_selected = (selected - mean[:, None]).square().sum(-1)
    fraction = centered_selected / (variance * values.shape[-1]).clamp_min(1e-20)
    return {
        "positions": values.shape[0],
        "selected_mean": selected.mean(0).cpu().tolist(),
        "selected_std": selected.std(0, unbiased=False).cpu().tolist(),
        "selected_rms": selected.square().mean(0).sqrt().cpu().tolist(),
        "selected_max_absolute": selected.abs().amax(0).cpu().tolist(),
        "mean_centered_variance_fraction_in_selected": float(fraction.mean()),
        "mean_std_ratio_if_selected_zeroed": float((zeroed_variance / variance.clamp_min(1e-20)).sqrt().mean()),
        "overall_rms": float(values.square().mean().sqrt()),
    }


class ReservationTrace:
    def __init__(self, model, selection, steps=(0, 50, 100, 150, 199)):
        self.steps = set(steps)
        self.step = 0
        self.active = False
        self.selection = selection
        self.rows = []
        self.handles = []
        score = model.structure_module.score_model
        self.handles.append(score.register_forward_pre_hook(self.start, with_kwargs=True))
        self.handles.append(score.register_forward_hook(self.finish))
        self.handles.append(score.atom_attention_encoder.register_forward_hook(self.encoder))
        self.handles.append(score.s_to_a_linear.register_forward_hook(self.conditioner))
        for index, layer in enumerate(score.token_transformer.layers):
            self.handles.append(layer.register_forward_pre_hook(self.before_layer(index)))
        self.handles.append(score.a_norm.register_forward_pre_hook(self.before_norm))

    def start(self, module, inputs, named):
        self.active = self.step in self.steps
        if self.active:
            features = named["feats"]
            self.token_type = features["mol_type"].reshape(-1)
            self.token_pad = features["token_pad_mask"].reshape(-1).bool()
            self.atom_type = torch.bmm(features["atom_to_token"].float(), features["mol_type"].float().unsqueeze(-1)).reshape(-1)
            self.atom_pad = features["atom_pad_mask"].reshape(-1).bool()

    def record(self, label, value, kind):
        if not self.active:
            return
        value = value.reshape(-1, value.shape[-1])
        slots = self.selection[f"{kind}_slots"]
        types, pad = ((self.token_type, self.token_pad) if kind == "token" else (self.atom_type, self.atom_pad))
        if len(value) != len(types):
            raise ValueError("Reservation traces require one diffusion sample")
        for name, mol_type in (("protein", 0), ("ligand", 3)):
            mask = (types == mol_type) & pad
            if not mask.any():
                continue
            self.rows.append({"step": self.step, "label": label, "kind": kind, "group": name,
                              **channel_statistics(value[mask], slots)})

    def encoder(self, module, inputs, output):
        self.record("atom_encoder_token_output", output[0], "token")
        self.record("atom_encoder_q", output[1], "atom")
        self.record("atom_encoder_c", output[2], "atom")

    def conditioner(self, module, inputs, output):
        self.record("single_conditioning_addition", output, "token")

    def before_layer(self, index):
        def observe(module, inputs):
            self.record(f"token_layer_{index}_input", inputs[0], "token")
        return observe

    def before_norm(self, module, inputs):
        self.record("final_token_norm_input", inputs[0], "token")

    def finish(self, module, inputs, output):
        self.step += 1

    def save(self, path):
        for handle in self.handles:
            handle.remove()
        path.write_text(json.dumps({
            "scope": "Read-only activation statistics. Counterfactual zeroing is calculated only in statistics and is never applied to model activations.",
            "atom_slots": self.selection["atom_slots"], "token_slots": self.selection["token_slots"],
            "diffusion_calls_observed": self.step, "rows": self.rows,
        }, indent=2) + "\n")
