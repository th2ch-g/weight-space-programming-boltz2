import sys
from pathlib import Path

import pytest
import torch
from boltz.model.modules.diffusionv2 import DiffusionModule
from boltz.model.modules.encodersv2 import AtomEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from program_reserved_channels import reserve_channels


@pytest.mark.parametrize(("scope", "preserve_condition_bias"), [("both", False), ("atom", False), ("token", False), ("token", True)])
def test_reserved_states_stay_zero_through_original_boltz_forward(scope, preserve_condition_bias):
    torch.manual_seed(41)
    model = DiffusionModule(
        token_s=24, atom_s=16, atom_encoder_depth=2, atom_encoder_heads=4,
        token_transformer_depth=2, token_transformer_heads=4,
        atom_decoder_depth=2, atom_decoder_heads=4,
    ).eval()
    encoder = AtomEncoder(16, 8, 24, 12, 32, 128, 388).eval()
    with torch.no_grad():
        for module in (model, encoder):
            for parameter in module.parameters():
                parameter.normal_(0, 0.1)
    prefixes = ("structure_module.score_model.", "diffusion_conditioning.atom_encoder.")
    state = {prefix + key: value for prefix, module in zip(prefixes, (model, encoder), strict=True) for key, value in module.state_dict().items()}
    original = {key: value.clone() for key, value in state.items()}
    edits, metadata = reserve_channels(state, count=4, scope=scope, preserve_condition_bias=preserve_condition_bias)
    assert all(torch.equal(state[key], value) for key, value in original.items())
    layer = model.token_transformer.layers[0]
    empty_state = torch.zeros(1, 3, 48)
    conditioning = torch.randn(1, 3, 48)
    mask = torch.ones(1, 3)
    bias = torch.zeros(1, 3, 3, 4)
    with torch.no_grad():
        normal = layer.adaln(empty_state, conditioning)
        original_condition_output = layer.pair_bias_attn(normal, bias, mask, normal)
    for prefix, module in zip(prefixes, (model, encoder), strict=True):
        module.load_state_dict({key.removeprefix(prefix): value for key, value in edits.items() if key.startswith(prefix)}, strict=False)
    atom_slots, token_slots = metadata["atom_slots"], metadata["token_slots"]
    if preserve_condition_bias:
        with torch.no_grad():
            normal = layer.adaln(empty_state, conditioning)
            preserved = layer.pair_bias_attn(normal, bias, mask, normal)
        retained = [index for index in range(48) if index not in token_slots]
        assert torch.equal(preserved[..., retained], original_condition_output[..., retained])
    observations = []

    def check_slots(slots):
        def hook(module, inputs, output):
            observations.append(output[..., slots].detach())
        return hook

    for module in (model.atom_attention_encoder.atom_encoder, model.atom_attention_decoder.atom_decoder):
        for layer in module.diffusion_transformer.layers:
            layer.register_forward_hook(check_slots(atom_slots))
    for layer in model.token_transformer.layers:
        layer.register_forward_hook(check_slots(token_slots))
    model.a_norm.register_forward_hook(check_slots(token_slots))
    features = {
        "ref_pos": torch.randn(1, 32, 3), "ref_charge": torch.zeros(1, 32),
        "ref_element": torch.randn(1, 32, 128), "ref_atom_name_chars": torch.randn(1, 32, 4, 64),
        "ref_space_uid": torch.zeros(1, 32, dtype=torch.long), "atom_pad_mask": torch.ones(1, 32),
        "atom_to_token": torch.eye(32)[None], "token_pad_mask": torch.ones(1, 32),
    }
    with torch.no_grad():
        q, c, _, to_keys = encoder(features, torch.randn(1, 32, 24), torch.randn(1, 32, 32, 12))
        assert torch.count_nonzero(q[..., atom_slots]) == 0
        assert torch.count_nonzero(c[..., atom_slots]) == 0
        result = model(
            torch.randn(1, 32, 24), torch.randn(1, 32, 24), torch.randn(1, 32, 3),
            torch.zeros(1), features,
            {"q": q, "c": c, "to_keys": to_keys,
             "atom_enc_bias": torch.zeros(1, 1, 32, 128, 8),
             "atom_dec_bias": torch.zeros(1, 1, 32, 128, 8),
             "token_trans_bias": torch.zeros(1, 32, 32, 8)},
        )
    assert len(observations) == 7
    assert all(torch.count_nonzero(value) == 0 for value in observations)
    assert torch.isfinite(result).all() and result.abs().max() > 1e-5
