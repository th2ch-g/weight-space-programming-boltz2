import numpy as np
import torch


class LayerTrace:
    def __init__(self, model, steps=(0, 50, 100, 125, 150, 175, 199)):
        self.model = model
        self.steps = set(steps)
        self.step = 0
        self.active = False
        self.arrays = {}
        self.frames = []
        self.handles = []
        score = model.structure_module.score_model
        self.readout = score.atom_attention_decoder.atom_feat_to_atom_pos_update
        self.handles.append(score.register_forward_pre_hook(self.start, with_kwargs=True))
        self.handles.append(score.register_forward_hook(self.finish, with_kwargs=True))
        self.handles.append(
            score.atom_attention_encoder.register_forward_hook(self.encoder, with_kwargs=True)
        )
        layers = score.atom_attention_decoder.atom_decoder.diffusion_transformer.layers
        for index, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self.before_layer(index)))
            self.handles.append(layer.register_forward_hook(self.after_layer(index)))

    def start(self, module, inputs, named):
        self.active = self.step in self.steps
        if not self.active:
            return
        feats = named["feats"]
        atom_type = torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].float().unsqueeze(-1)
        ).squeeze(-1)
        self.mask = ((atom_type == 3) & feats["atom_pad_mask"].bool()).reshape(-1)
        self.atom_count = feats["ref_pos"].shape[1]
        sigma = (named["times"].float() * 4).exp() * self.model.structure_module.sigma_data
        sigma = sigma.reshape(-1, 1, 1)
        x = named["r_noisy"].float() / self.model.structure_module.c_in(sigma)
        self.skip = (self.model.structure_module.c_skip(sigma) * x).reshape(-1, 3)[self.mask]
        self.scale = self.model.structure_module.c_out(sigma)[0, 0, 0]
        self.frames.append((self.step, "noisy", x.reshape(-1, 3)[self.mask].cpu().numpy()))
        if "s_inputs" not in self.arrays:
            for name in ("s_inputs", "s_trunk"):
                self.arrays[name] = named[name][0].float().cpu().numpy()
            for name in ("ref_pos", "mol_type", "atom_to_token", "atom_pad_mask"):
                self.arrays[name] = feats[name][0].float().cpu().numpy()
            for name in ("q", "c"):
                self.arrays[f"conditioning_{name}"] = (
                    named["diffusion_conditioning"][name][0].float().cpu().numpy()
                )

    def project(self, label, q):
        if not self.active:
            return
        q = q.reshape(-1, q.shape[-1])[self.mask]
        with torch.autocast("cuda", enabled=False):
            xyz = self.skip + self.scale * self.readout(q.float())
        self.frames.append((self.step, label, xyz.float().cpu().numpy()))

    def encoder(self, module, inputs, named, result):
        self.project("encoder_output", result[1])

    def before_layer(self, index):
        def observe(module, inputs):
            self.project(f"decoder_{index}_input", inputs[0])

        return observe

    def after_layer(self, index):
        def observe(module, inputs, result):
            self.project(f"decoder_{index}_output", result)

        return observe

    def finish(self, module, inputs, named, result):
        if self.active:
            xyz = self.skip + self.scale * result.reshape(-1, 3)[self.mask]
            self.frames.append((self.step, "score_output", xyz.float().cpu().numpy()))
        self.step += 1

    def save(self, path):
        for handle in self.handles:
            handle.remove()
        if not self.frames:
            return
        np.savez_compressed(
            path,
            **self.arrays,
            steps=np.array([frame[0] for frame in self.frames]),
            labels=np.array([frame[1] for frame in self.frames]),
            coordinates=np.stack([frame[2] for frame in self.frames]),
        )
