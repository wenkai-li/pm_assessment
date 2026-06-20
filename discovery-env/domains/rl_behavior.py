"""Domain 5 — Learned-policy and RL behavior. (Specified here; backend not implemented in this POC.)

System S: a trained policy that relies on a hidden learned quirk.
Hidden H: a learned heuristic or reward-hack tied to a specific observation feature, randomized.
Tools: run rollouts from chosen states, perturb the environment or observations, intervene on
  observation features (behavioral patching), read policy activations.
Contract:
  necessity    -> mask the feature; predict the behavior changes to a value.
  sufficiency  -> the behavior is reproduced by that feature alone.
  directional  -> predict the behavior shift under a graded perturbation of the feature.
  extrapolation-> predict under held-out environment perturbations.
Ground truth is known because the env constructs the policy's environment, which drives calibration.
Researcher scenario: understanding why a policy behaves a certain way, finding reward-hacks.
"""
from domains.base import Domain


class RLBehaviorDomain(Domain):
    name = "rl_behavior"
    _SPEC = "RL-behavior backend not implemented in this POC. See GENERAL_BENCHMARK_DESIGN, Domain 5."

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
