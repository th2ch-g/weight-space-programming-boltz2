import torch


def reserve_channels(state, count=18, scope="both", preserve_condition_bias=False):
    """Reserve residual channels using only fixed checkpoint-derived rankings."""
    score = "structure_module.score_model."
    atom = "diffusion_conditioning.atom_encoder."
    encoder = score + "atom_attention_encoder."
    decoder = score + "atom_attention_decoder."
    coordinate = decoder + "atom_feat_to_atom_pos_update."
    atom_readers = [
        state[encoder + "atom_to_token_trans.0.weight"].double(),
        state[coordinate + "1.weight"].double()
        * state[coordinate + "0.weight"].double()[None],
    ]
    atom_scores = sum(
        reader.square().sum(0) / reader.square().sum().clamp_min(1e-30)
        for reader in atom_readers
    )
    token_reader = (state[decoder + "a_to_q_trans.weight"].double()
                    * state[score + "a_norm.weight"].double()[None])
    token_scores = token_reader.square().sum(0)
    if not 1 <= count < min(atom_scores.numel(), token_scores.numel()):
        raise ValueError("Invalid reservation size")
    if scope not in {"both", "atom", "token"}:
        raise ValueError("Unknown reservation scope")
    if preserve_condition_bias and scope != "token":
        raise ValueError("Condition-bias preservation is only defined for the token stream")
    atom_slots = torch.argsort(atom_scores, stable=True)[:count].sort().values
    token_slots = torch.argsort(token_scores, stable=True)[:count].sort().values
    if scope == "token":
        atom_slots = atom_slots[:0]
    if scope == "atom":
        token_slots = token_slots[:0]
    edits = {}

    def erase(key, axis, slots):
        if not len(slots):
            return
        if key not in edits:
            edits[key] = state[key].clone()
        edits[key].index_fill_(axis, slots, 0)

    for suffix in ("embed_atom_features.weight", "embed_atom_features.bias", "s_to_c_trans.1.weight"):
        erase(atom + suffix, 0, atom_slots)
    for suffix in ("c_to_p_trans_q.1.weight", "c_to_p_trans_k.1.weight"):
        erase(atom + suffix, 1, atom_slots)
    erase(encoder + "r_to_q_trans.weight", 0, atom_slots)
    erase(encoder + "atom_to_token_trans.0.weight", 1, atom_slots)
    erase(encoder + "atom_to_token_trans.0.weight", 0, token_slots)
    erase(score + "s_to_a_linear.1.weight", 0, token_slots)
    erase(decoder + "a_to_q_trans.weight", 1, token_slots)
    erase(decoder + "a_to_q_trans.weight", 0, atom_slots)
    erase(coordinate + "1.weight", 1, atom_slots)
    for suffix in ("weight", "bias"):
        erase(score + "a_norm." + suffix, 0, token_slots)
        erase(coordinate + "0." + suffix, 0, atom_slots)

    groups = (
        (encoder + "atom_encoder.diffusion_transformer.layers.", atom_slots, True),
        (decoder + "atom_decoder.diffusion_transformer.layers.", atom_slots, True),
        (score + "token_transformer.layers.", token_slots, False),
    )
    layer_counts = []
    for prefix, slots, reserve_condition in groups:
        indices = sorted({int(key[len(prefix):].split(".")[0]) for key in state if key.startswith(prefix)})
        if not indices or indices != list(range(len(indices))):
            raise ValueError("Unexpected transformer layer numbering")
        layer_counts.append(len(indices))
        for index in indices:
            layer = prefix + str(index) + "."
            if any(key.startswith(layer + "post_lnorm.") for key in state):
                raise ValueError("Residual reservation requires an identity post normalization")
            if preserve_condition_bias and len(slots):
                for branch in ("adaln.", "transition.adaln."):
                    erase(layer + branch + "s_scale.weight", 0, slots)
                    key = layer + branch + "s_scale.bias"
                    edits[key] = state[key].clone()
                    edits[key][slots] = -16384
            else:
                for projection in ("proj_q", "proj_k", "proj_v", "proj_g"):
                    erase(layer + "pair_bias_attn." + projection + ".weight", 1, slots)
            erase(layer + "pair_bias_attn.proj_o.weight", 0, slots)
            if not preserve_condition_bias:
                erase(layer + "transition.swish_gate.0.weight", 1, slots)
                erase(layer + "transition.a_to_b.weight", 1, slots)
            erase(layer + "transition.b_to_a.weight", 0, slots)
            if reserve_condition:
                for branch in ("adaln.", "transition.adaln."):
                    erase(layer + branch + "s_norm.weight", 0, slots)
                    erase(layer + branch + "s_scale.weight", 1, slots)
                    erase(layer + branch + "s_bias.weight", 1, slots)
                for branch in ("output_projection_linear.", "output_projection.0.", "transition.output_projection.0."):
                    erase(layer + branch + "weight", 1, slots)
    metadata = {
        "scope": scope,
        "preserve_condition_bias": preserve_condition_bias,
        "atom_slots": atom_slots.tolist(),
        "token_slots": token_slots.tolist(),
        "atom_ranking": "Sum of normalized squared column norms of atom-to-token and gamma-weighted final coordinate projections.",
        "token_ranking": "Squared column norms of the gamma-weighted token-to-atom projection.",
        "atom_selected_scores": atom_scores[atom_slots].tolist(),
        "token_selected_scores": token_scores[token_slots].tolist(),
        "layer_counts_encoder_decoder_token": layer_counts,
        "limitation": "Zeroing channels changes LayerNorm means and variances. This control is not function preserving and does not repair chemistry.",
    }
    return edits, metadata
