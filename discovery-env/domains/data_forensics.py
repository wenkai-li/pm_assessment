"""Domain 3 — Data forensics and attribution. (Specified here; backend not implemented in this POC.)

System S: a model trained on a dataset that carries a hidden property.
Hidden H: a planted spurious correlation, a contamination, or a subpopulation imbalance, randomized.
Tools: query the model, sample and inspect the data, train a probe, apply a counterfactual data
  intervention (remove or reweight a subset) and retrain a cheap surrogate, evaluate on held-out
  distributions.
Contract:
  necessity    -> remove the subpopulation; predict the behavior drops to a value on held-out data.
  sufficiency  -> the behavior is reproduced by that subpopulation alone.
  directional  -> predict performance on a counterfactual distribution where the spurious feature is
                  decorrelated.
  extrapolation-> predict on a held-out distribution the agent never evaluated.
Ground truth is known because the env plants the property, which drives the calibration gate.
Researcher scenario: spurious-correlation discovery, dataset debugging, contamination forensics.
"""
from domains.base import Domain


class DataForensicsDomain(Domain):
    name = "data_forensics"
    _SPEC = "Data-forensics backend not implemented in this POC. See GENERAL_BENCHMARK_DESIGN, Domain 3."

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
