"""The model-internals domain (the worked instance), adapting the modular-addition Fourier-circuit
code to the shared Domain interface."""
import calibration as calib
from contract import grade_ladder
from domains.base import Domain
from grader import _heldout, check_prediction
from interventions import make_batch
from io_utils import load_model
from reference_solver import build_reference_contract


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
        """Grade a committed contract along the derivation ladder (shared grade_ladder)."""
        return grade_ladder(self, instance, contract, heldout, probes)

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
