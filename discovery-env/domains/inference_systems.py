"""Domain 4 — Inference and systems behavior. (Specified here; backend not implemented in this POC.)

System S: a serving stack that exhibits an anomaly under some condition.
Hidden H: a hidden configuration interaction, randomized per instance (batching and padding, a
  quantized layer, a KV-cache eviction).
Tools: run the server at chosen configs and inputs, read intermediate tensors, toggle config knobs,
  compare batched against single, checkpoint/restore.
Contract:
  necessity    -> set the right knob; the anomaly disappears; predict the metric on held-out inputs.
  sufficiency  -> the anomaly is reproduced by that knob alone.
  directional  -> predict how the error scales with batch size, sequence length, or quantization bits.
  extrapolation-> predict on a held-out input regime.
Ground truth is known because the env injects the configuration fault, which drives the calibration.
Researcher scenario: inference debugging, systems forensics, regression triage.
"""
from domains.base import Domain


class InferenceSystemsDomain(Domain):
    name = "inference_systems"
    _SPEC = "Inference-systems backend not implemented in this POC. See GENERAL_BENCHMARK_DESIGN, Domain 4."

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
