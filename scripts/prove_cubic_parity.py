import argparse
import itertools
import json
from pathlib import Path

import torch
from boltz.model.modules.transformersv2 import ConditionedTransitionBlock


def determinant_block(dtype=torch.float64):
    block = ConditionedTransitionBlock(128, 128).to(dtype=dtype).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        gate_weight = block.swish_gate[0].weight
        half = gate_weight.shape[0] // 2
        for term, order in enumerate(itertools.permutations(range(3))):
            inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
            sign = (-1) ** inversions
            for pair, gate_sign in enumerate((1, -1)):
                neuron = 2 * term + pair
                gate_weight[half + neuron, order[0]] = gate_sign
                gate_weight[neuron, 3 + order[1]] = 1
                block.a_to_b.weight[neuron, 6 + order[2]] = 1
                block.b_to_a.weight[0, neuron] = 2 * sign * gate_sign
    return block


def encode(vectors):
    flat = vectors.flatten(-2)
    hidden = torch.zeros((*flat.shape[:-1], 128), dtype=flat.dtype, device=flat.device)
    hidden[..., :9] = flat
    hidden[..., 9:18] = -flat
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/cubic_parity_proof.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    generator = torch.Generator().manual_seed(20261103)
    vectors = torch.randn(512, 3, 3, generator=generator, dtype=torch.float64)
    matrices = torch.randn(512, 3, 3, generator=generator, dtype=torch.float64)
    rotations, _ = torch.linalg.qr(matrices)
    rotations[:, :, 0] *= torch.linalg.det(rotations)[:, None]
    def observe(block, value, mixed=False):
        hidden = encode(value)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
            return block(hidden, torch.zeros_like(hidden))[:, 0].double()

    results = []
    for dtype, mixed, tolerance in (
        (torch.float64, False, 1e-10),
        (torch.float32, False, 2e-5),
        (torch.float32, True, None),
    ):
        block = determinant_block(dtype)
        prediction = observe(block, vectors.to(dtype), mixed)
        scale = 0.5 / torch.sqrt(2 * vectors.square().sum((1, 2)) / 128 + block.adaln.a_norm.eps)
        expected = torch.linalg.det(vectors) * scale**3
        proper = observe(block, (vectors @ rotations).to(dtype), mixed)
        reflected_vectors = vectors.clone()
        reflected_vectors[:, :, 0] *= -1
        reflected = observe(block, (reflected_vectors @ rotations).to(dtype), mixed)
        errors = {
            "determinant_formula_max_absolute_error": float((prediction - expected).abs().max()),
            "proper_rotation_max_absolute_error": float((prediction - proper).abs().max()),
            "reflection_sign_max_absolute_error": float((prediction + reflected).abs().max()),
        }
        if tolerance is not None and max(errors.values()) > tolerance:
            raise ValueError(f"Constructive parity identity failed for {dtype}: {errors}")
        results.append({
            "dtype": str(dtype),
            "cpu_bfloat16_autocast": mixed,
            "errors": errors,
            "sign_disagreement_count": int((prediction.sign() != expected.sign()).sum()),
            "proper_rotation_sign_disagreement_count": int((proper.sign() != expected.sign()).sum()),
            "reflected_sign_disagreement_count": int((reflected.sign() != -expected.sign()).sum()),
        })
    result = {
        "scope": "Synthetic construction in the unmodified Boltz ConditionedTransitionBlock class.",
        "method": "Twelve paired SiLU neurons implement six determinant monomials using SiLU(x)-SiLU(-x)=x. Signed input channel pairs make LayerNorm centering zero; normalization contributes a positive common scale.",
        "trained_or_optimized": False,
        "boltz_checkpoint_modified": False,
        "samples": len(vectors),
        "input_channels": 18,
        "cubic_neurons": 12,
        "precision_results": results,
        "limitation": "This proves only a representable scalar primitive. CPU autocast is not a CUDA validation. The construction does not supply three neighbor vectors inside the pretrained encoder, connect the scalar to the decoder, preserve protein predictions, or demonstrate any ct209 repair.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
