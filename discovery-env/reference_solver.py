"""The environment's own ground truth. Because the environment trained the model, it can read off
the key frequencies directly and build the reference causal-claim contract used for calibration.
A solver agent must rediscover this from the outside, with no access to these functions.
"""
import torch

from fourier import make_fourier_basis
from grader import algorithm_corr, specificity_curve
from interventions import ablate, make_batch, phase_shift


def find_key_frequencies(model, p, rel_threshold=0.3):
    F = make_fourier_basis(p)
    coeffs = F @ model.W_E.data[:p]  # (p, d_model)
    norms = torch.zeros(p // 2 + 1)
    for k in range(1, p // 2 + 1):
        norms[k] = (coeffs[2 * k - 1].norm() ** 2 + coeffs[2 * k].norm() ** 2).sqrt()
    cutoff = rel_threshold * norms.max()
    return [k for k in range(1, p // 2 + 1) if norms[k] >= cutoff]


def build_reference_contract(model, p, heldout):
    """Construct a contract whose predictions equal the measured effects, so it should score ~1."""
    K = find_key_frequencies(model, p)
    tokens, targets = make_batch(heldout, p)

    nec = ablate(model, tokens, targets, K, p, mode="remove")
    suf = ablate(model, tokens, targets, K, p, mode="keep")
    probe = sorted(set(K + [k for k in range(1, p // 2 + 1) if k not in K][:5]))
    spec = specificity_curve(model, tokens, targets, probe, p)
    corr = algorithm_corr(model, tokens, p, K)
    k0 = K[0]
    _, _, rate = _phase_rate(model, tokens, p, k0)

    return {
        "components": {"basis": "fourier", "elements": K},
        "predictions": [
            {"id": "necessity", "predicted": round(nec, 4), "tol": 0.03},
            {"id": "sufficiency", "predicted": round(suf, 4), "tol": 0.03},
            {"id": "specificity", "probe": probe, "predicted": [round(x, 4) for x in spec], "tol": 0.05},
            {"id": "algorithm", "predicted": round(corr, 4), "tol": 0.05},
            {"id": "phase_shift", "freq": k0, "shift": 1, "predicted": round(rate, 4), "tol": 0.1},
        ],
    }


def _phase_rate(model, tokens, p, freq, shift=1):
    logits, predicted = phase_shift(model, tokens, p, freq, shift)
    rate = (logits.argmax(-1) == predicted).float().mean().item()
    return logits, predicted, rate
