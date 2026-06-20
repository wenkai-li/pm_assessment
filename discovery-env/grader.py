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
    """Pearson correlation between the model logits and the Fourier-clock formula on these inputs."""
    a, b = tokens[:, 0], tokens[:, 1]
    s = (a + b).float()
    c = torch.arange(p).float()
    ang = 2 * math.pi * (s[:, None] - c[None, :]) / p  # (n, p)
    pred = torch.zeros(len(tokens), p)
    for k in K:
        pred = pred + torch.cos(k * ang)
    actual = model_logits(model, tokens)[:, :p]  # drop the '=' token logit; answers are 0..p-1
    x, y = pred.flatten(), actual.flatten()
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y / (x.norm() * y.norm() + 1e-9)).item()


def specificity_curve(model, tokens, targets, probe, p):
    """Accuracy after removing each single probe frequency on its own."""
    return [ablate(model, tokens, targets, [k], p, mode="remove") for k in probe]


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
        actual = specificity_curve(model, tokens, targets, pred["probe"], p)
        err = sum(abs(x - y) for x, y in zip(actual, pred["predicted"])) / len(actual)
        passed = err <= pred["tol"]
    elif pid == "algorithm":
        actual = algorithm_corr(model, tokens, p, K)
        passed = (actual >= 0.9) and abs(actual - pred["predicted"]) <= pred["tol"]
    elif pid == "phase_shift":
        logits, predicted = phase_shift(model, tokens, p, pred["freq"], pred.get("shift", 1))
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
