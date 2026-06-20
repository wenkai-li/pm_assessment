"""The Domain interface. The benchmark generator and the grader depend ONLY on this interface,
which is what makes The Discovery a benchmark (one verifier over many domains) rather than one task.

Implement this interface to add a research-discovery domain. The shared prediction-contract grader
in contract.py, the calibration step, and the scoring are reused unchanged.
"""
from abc import ABC, abstractmethod


class Domain(ABC):
    name = "abstract"

    @abstractmethod
    def build_instance(self, seed):
        """Construct a system S with a randomly placed hidden structure H. Return an instance dict
        (the env knows H, which is what makes per-instance calibration possible)."""

    @abstractmethod
    def tools(self):
        """Return the action space exposed to the agent for this domain (a set/dict of tool names)."""

    @abstractmethod
    def heldout(self, instance, seed):
        """Return fresh conditions the agent never saw, on which the grader checks predictions."""

    @abstractmethod
    def execute(self, pred, instance, structure, heldout):
        """Run one prediction-contract claim's intervention. Return (actual_value, passed: bool)."""

    @abstractmethod
    def structure_size(self, structure):
        """Size of the claimed structure, for the minimality bonus."""

    @abstractmethod
    def max_structure(self, instance):
        """Normalizer for the minimality bonus."""

    @abstractmethod
    def reference_contract(self, instance):
        """The env's own ground-truth contract (it knows H). Should score high in calibration."""

    @abstractmethod
    def calibration_contracts(self, instance):
        """Dict name -> contract: the reference plus a bank of hacks, for the no-FP/no-FN gate."""
