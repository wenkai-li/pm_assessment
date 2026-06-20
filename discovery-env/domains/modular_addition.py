"""The model-internals domain (the worked instance), adapting the modular-addition Fourier-circuit
code to the shared Domain interface."""
import calibration as calib
from domains.base import Domain
from grader import _heldout, check_prediction
from interventions import make_batch
from io_utils import load_model
from reference_solver import build_reference_contract


def prefix_len(flags):
    """Length of the longest leading run of truthy flags (the depth-gated prefix)."""
    n = 0
    for f in flags:
        if f:
            n += 1
        else:
            break
    return n


class ModularAdditionDomain(Domain):
    name = "modular_addition"

    # The mechanistic derivation as a dependency-ordered ladder, shallow to deep. Each rung can only
    # hold if the rungs before it hold: the closed-form algorithm (rung 4) cannot be correct without
    # the right key frequencies (rungs 1-2), and the directional phase shift (rung 5) uses a key
    # frequency, so verified depth along this ladder measures how deep into the mechanism the account
    # actually reaches. Committing after rungs 1-2 (naming the components without deriving how they
    # compute the answer) is the shallow, premature account this benchmark is built to catch.
    DERIVATION_LADDER = ("necessity", "sufficiency", "specificity", "algorithm", "phase_shift")

    def __init__(self, ckpt=None):
        self.ckpt = ckpt

    def required_depth(self, instance):
        return len(self.DERIVATION_LADDER)

    def grade_episode(self, instance, contract, heldout, probes=0):
        """Grade a committed contract along the derivation ladder and return a result dict for the
        depth-calibrated judge (judge.score_episode). commit_depth is how deep the agent claimed
        (leading rungs it supplied); correct_depth is how deep its claims actually hold on held-out
        inputs; solved means the full mechanism is verified."""
        preds = {p["id"]: p for p in contract["predictions"]}
        ladder = self.DERIVATION_LADDER

        commit_depth = prefix_len(rung in preds for rung in ladder)

        pass_flags = []
        for rung in ladder[:commit_depth]:
            _, passed = self.execute(preds[rung], instance, contract["components"], heldout)
            pass_flags.append(passed)
        correct_depth = prefix_len(pass_flags)
        required = len(ladder)

        return {"solved": correct_depth == required, "correct_depth": correct_depth,
                "commit_depth": commit_depth, "required": required, "probes": probes}

    def build_instance(self, seed):
        # The POC trains checkpoints offline (train_modular_addition.py); here we load one.
        model = load_model(self.ckpt) if self.ckpt else None
        p = model.cfg.p if model is not None else 113
        return {"model": model, "p": p, "seed": seed,
                "symptom": "Maps (a, b) -> (a + b) mod p at ~100% accuracy. Recover the mechanism."}

    def tools(self):
        return {"run", "cache", "ablate", "patch", "phase_shift", "fourier_basis",
                "checkpoint", "restore"}

    def heldout(self, instance, seed=12345):
        return _heldout(instance["p"], seed=seed)

    def execute(self, pred, instance, structure, heldout):
        model, p = instance["model"], instance["p"]
        K = list(structure["elements"])
        tokens, targets = make_batch(heldout, p)
        return check_prediction(model, p, K, pred, tokens, targets)

    def structure_size(self, structure):
        return len(structure["elements"])

    def max_structure(self, instance):
        return instance["p"] // 2

    def reference_contract(self, instance):
        return build_reference_contract(instance["model"], instance["p"], self.heldout(instance))

    def calibration_contracts(self, instance):
        return calib.make_contracts(instance["model"], instance["p"], self.heldout(instance))
