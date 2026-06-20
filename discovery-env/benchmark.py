"""Run the generic prediction-contract grader through the Domain interface.

This demonstrates that the verifier is domain-agnostic: the same contract.grade runs for any domain
in the registry. modular_addition is implemented; the other domains are specified stubs.

  python3 benchmark.py --list
  python3 benchmark.py --domain modular_addition --ckpt checkpoints/seed_0.pt
"""
import argparse

from contract import grade
from domains import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="modular_addition", choices=list(REGISTRY))
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
    print(f"domain = {domain.name}")
    print(f"{'contract':<14}{'score':<8}")
    print("-" * 22)
    for name, contract in domain.calibration_contracts(instance).items():
        score, _ = grade(domain, instance, contract, domain.heldout(instance))
        print(f"{name:<14}{score:<8.3f}")


if __name__ == "__main__":
    main()
