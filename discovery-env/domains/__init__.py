"""Domain registry. modular_addition is implemented (worked instance); the other four are specified
as stubs that share the same interface (see GENERAL_BENCHMARK_DESIGN for their full specs)."""
from domains.data_forensics import DataForensicsDomain
from domains.inference_systems import InferenceSystemsDomain
from domains.modular_addition import ModularAdditionDomain
from domains.rl_behavior import RLBehaviorDomain
from domains.training_dynamics import TrainingDynamicsDomain

REGISTRY = {
    "modular_addition": ModularAdditionDomain,   # implemented (interpretability worked instance)
    "data_forensics": DataForensicsDomain,       # implemented (second worked instance, pure numpy)
    "training_dynamics": TrainingDynamicsDomain,  # specified stub
    "inference_systems": InferenceSystemsDomain,  # specified stub
    "rl_behavior": RLBehaviorDomain,              # specified stub
}
