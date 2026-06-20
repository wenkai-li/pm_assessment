"""The model-internals domain (the worked instance), adapting the modular-addition Fourier-circuit
code to the shared Domain interface."""
import calibration as calib
from domains.base import Domain
from grader import _heldout, check_prediction
from interventions import make_batch
from io_utils import load_model
from reference_solver import build_reference_contract


class ModularAdditionDomain(Domain):
    name = "modular_addition"

    def __init__(self, ckpt=None):
        self.ckpt = ckpt

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
