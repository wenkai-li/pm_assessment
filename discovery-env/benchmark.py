"""Run the depth-calibrated judge over any domain through the shared interface.

This demonstrates that the verifier is domain-agnostic: the same derivation-ladder grading
(contract.grade_ladder) and the same per-episode judge (judge.score_episode) score calibration
contracts for any domain that defines DERIVATION_LADDER. modular_addition and data_forensics are
implemented; the other domains are specified stubs.

  python3 benchmark.py --list
  python3 benchmark.py --domain data_forensics
  python3 benchmark.py --domain modular_addition --ckpt checkpoints/seed_0.pt
"""
import argparse

from domains import REGISTRY
from judge import score_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="data_forensics", choices=list(REGISTRY))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k in REGISTRY:
            print(k)
        return

    DomainCls = REGISTRY[args.domain]
    domain = DomainCls(ckpt=args.ckpt) if args.domain == "modular_addition" else DomainCls()
    instance = domain.build_instance(seed=0)
    heldout = domain.heldout(instance)

    print(f"domain = {domain.name}   required_depth = {len(domain.DERIVATION_LADDER)}")
    print(f"{'contract':<20}{'score':<10}{'correct_depth':<15}{'solved'}")
    print("-" * 52)
    for name, contract in domain.calibration_contracts(instance).items():
        result = domain.grade_episode(instance, contract, heldout)
        j = score_episode(result)
        print(f"{name:<20}{j.score:<10.3f}{j.correct_depth:<15}{j.solved}")


if __name__ == "__main__":
    main()
