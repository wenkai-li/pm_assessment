"""Score a causal-claim contract by running every intervention on held-out inputs.

No language-model judging: each prediction is checked against an effect the grader measures itself.
A contract scores well only if the claimed components are necessary and sufficient, the per-frequency
specificity pattern matches, the closed-form algorithm reproduces the logits, and the directional
phase-shift moves the answer as predicted. A smaller correct component set scores higher.
"""
import argparse
import json
import math

import torch

from interventions import ablate, make_batch, phase_shift, model_logits


def algorithm_corr(model, tokens, p, K):
    """How well the claimed key frequencies reconstruct the model logits.

    A real grokked model reads the answer out as a weighted sum of cos(2*pi*k*(a+b-c)/p) over its key
    frequencies, but the weights are not equal, so an equal-weight formula correlates poorly. This
    fits the best non-negative-free amplitudes for the claimed frequencies by least squares and returns
    the correlation of that best-fit reconstruction with the actual logits. The true frequency set
    reconstructs the logits well; a wrong or incomplete set does not, so the test stays discriminating
    while no longer penalizing the true mechanism for the idealized equal-weight assumption."""
    a, b = tokens[:, 0], tokens[:, 1]
    s = (a + b).float()
    c = torch.arange(p).float()
    ang = 2 * math.pi * (s[:, None] - c[None, :]) / p  # (n, p)
    cols = [torch.ones(len(tokens), p).flatten()]  # constant term
    for k in K:
        cols.append(torch.cos(k * ang).flatten())
        cols.append(torch.sin(k * ang).flatten())
    if len(cols) == 1:
        return 0.0
    design = torch.stack(cols, dim=1)  # (n*p, 1 + 2|K|)
    actual = model_logits(model, tokens)[:, :p].flatten()  # answers are 0..p-1
    coef, *_ = torch.linalg.lstsq(design, actual.unsqueeze(1))
    recon = (design @ coef).squeeze(1)
    x = recon - recon.mean()
    y = actual - actual.mean()
    return (x @ y / (x.norm() * y.norm() + 1e-9)).item()


def specificity_curve(model, tokens, targets, probe, p):
    """Accuracy after removing each single probe frequency on its own."""
    return [ablate(model, tokens, targets, [k], p, mode="remove") for k in probe]


def _freq_is_real(model, p, k, rel_threshold=0.15):
    """True if frequency k carries real energy in the number-token embedding, relative to the
    strongest frequency. Used for the minimality check, which rejects claiming inert frequencies."""
    import math as _m
    from fourier import make_fourier_basis
    F = make_fourier_basis(p)
    coeffs = F @ model.W_E.data[:p]
    norms = []
    for j in range(1, p // 2 + 1):
        norms.append((coeffs[2 * j - 1].norm() ** 2 + coeffs[2 * j].norm() ** 2).sqrt().item())
    target = (coeffs[2 * k - 1].norm() ** 2 + coeffs[2 * k].norm() ** 2).sqrt().item()
    return target >= rel_threshold * max(norms)


def check_prediction(model, p, K, pred, tokens, targets):
    """Execute one contract claim and return (actual_value, passed). Shared by the MI grader and by
    the generic benchmark grader in contract.py via the modular-addition domain backend."""
    chance = 1.0 / p
    pid = pred["id"]
    if pid == "necessity":
        actual = ablate(model, tokens, targets, K, p, mode="remove")
        passed = (actual <= chance + 0.03) and abs(actual - pred["predicted"]) <= pred["tol"]
    elif pid == "sufficiency":
        actual = ablate(model, tokens, targets, K, p, mode="keep")
        passed = (actual >= 0.9) and abs(actual - pred["predicted"]) <= pred["tol"]
    elif pid == "specificity":
        # Match the predicted per-frequency curve, AND enforce minimality: every claimed frequency must
        # be one the model actually represents, that is, carry real energy in the embedding. Inert
        # frequencies carry almost none, so this rejects claiming the whole frequency set while still
        # accepting a redundant true set whose individual frequencies do not each move accuracy alone.
        actual = specificity_curve(model, tokens, targets, pred["probe"], p)
        err = sum(abs(x - y) for x, y in zip(actual, pred["predicted"])) / len(actual)
        minimal = all(_freq_is_real(model, p, k) for k in K)
        passed = (err <= pred["tol"]) and minimal
    elif pid == "algorithm":
        # A real grokked model's logits are reconstructed by its key frequencies to about 0.8, not 1.0,
        # because higher-order terms carry the rest; the floor reflects that while staying well above
        # what a wrong or incomplete frequency set reconstructs.
        actual = algorithm_corr(model, tokens, p, K)
        passed = (actual >= 0.75) and abs(actual - pred["predicted"]) <= pred["tol"]
    elif pid == "phase_shift":
        # Rotate all claimed key frequencies coherently so the answer moves by `shift`; rotating one
        # frequency among several does not move the argmax on a real model.
        logits, predicted = phase_shift(model, tokens, p, K, pred.get("shift", 1))
        actual = (logits.argmax(-1) == predicted).float().mean().item()
        passed = (actual >= 0.8) and abs(actual - pred["predicted"]) <= pred["tol"]
    else:
        actual, passed = None, False
    return actual, passed


def _valid_freqs(freqs, p):
    return all(isinstance(k, int) and 1 <= k <= p // 2 for k in freqs)


def grade(model, claim, p, heldout, verbose=False):
    K = list(claim["components"]["elements"])
    tokens, targets = make_batch(heldout, p)
    checks, details = {}, {}

    # Step 2 of the judge: reject malformed / illegal specifications instead of crashing.
    probe_freqs = next((pr.get("probe", []) for pr in claim["predictions"] if pr["id"] == "specificity"), [])
    phase_freqs = [pr["freq"] for pr in claim["predictions"] if pr["id"] == "phase_shift"]
    if not _valid_freqs(K + list(probe_freqs) + phase_freqs, p):
        if verbose:
            print(json.dumps({"error": "illegal frequency in claim", "score": 0.0}, indent=2))
        return 0.0, {"checks": {}, "details": {}, "gate": False, "error": "illegal_freq"}

    for pred in claim["predictions"]:
        actual, passed = check_prediction(model, p, K, pred, tokens, targets)
        checks[pred["id"]] = passed
        details[pred["id"]] = actual

    gate = checks.get("necessity", False) and checks.get("sufficiency", False)
    mean_pass = sum(checks.values()) / max(len(checks), 1)
    min_factor = max(0.0, 1.0 - len(K) / (0.5 * p))
    score = mean_pass * min_factor if gate else 0.2 * mean_pass
    if verbose:
        print(json.dumps({"checks": checks, "details": details, "gate": gate, "score": score}, indent=2))
    return score, {"checks": checks, "details": details, "gate": gate, "min_factor": min_factor}


def _heldout(p, n=400, seed=12345):
    g = torch.Generator().manual_seed(seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    idx = torch.randperm(len(pairs), generator=g)[:n]
    return [pairs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--claim", required=True)
    args = ap.parse_args()
    from io_utils import load_model

    model = load_model(args.ckpt)
    with open(args.claim) as f:
        claim = json.load(f)
    score, info = grade(model, claim, model.cfg.p, _heldout(model.cfg.p), verbose=True)
    print(f"SCORE = {score:.3f}")


if __name__ == "__main__":
    main()
