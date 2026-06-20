"""Domain 2 — Training dynamics. (Specified here; backend not implemented in this POC.)

System S: a family of short training runs whose final metric follows a constructed law.
Hidden H: a planted relationship, randomized per instance, for example an emergent jump at a hidden
  threshold, a scaling exponent, or a hyperparameter interaction.
Tools: launch short training runs at chosen configs, read loss curves, checkpoint/restore.
Contract:
  necessity    -> intervene on the driving factor; predict the metric collapses.
  sufficiency  -> keep only that factor; predict the metric is preserved.
  extrapolation-> predict the metric at a held-out config the agent never ran (only the true law
                  extrapolates; a fitted curve does not).
  directional  -> predict how the threshold moves under a knob.
Ground truth is known because the env constructs the law, which drives the calibration gate.
Researcher scenario: discovering scaling laws, drivers of emergence, hyperparameter interactions.
"""
from domains.base import Domain


class TrainingDynamicsDomain(Domain):
    name = "training_dynamics"
    _SPEC = "Training-dynamics backend not implemented in this POC. See GENERAL_BENCHMARK_DESIGN, Domain 2."

    def build_instance(self, seed):
        raise NotImplementedError(self._SPEC)

    def tools(self):
        raise NotImplementedError(self._SPEC)

    def heldout(self, instance, seed=0):
        raise NotImplementedError(self._SPEC)

    def execute(self, pred, instance, structure, heldout):
        raise NotImplementedError(self._SPEC)

    def structure_size(self, structure):
        raise NotImplementedError(self._SPEC)

    def max_structure(self, instance):
        raise NotImplementedError(self._SPEC)

    def reference_contract(self, instance):
        raise NotImplementedError(self._SPEC)

    def calibration_contracts(self, instance):
        raise NotImplementedError(self._SPEC)
