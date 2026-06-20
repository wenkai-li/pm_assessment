"""The headline demonstration: build the reference contract and several wrong / hacking contracts,
grade them all, and print the table that shows no false positives and no false negatives.

This is what turns the no-FP / no-FN claim into a measured result instead of an assertion.
"""
import argparse
import copy

from grader import _heldout, grade
from io_utils import load_model
from reference_solver import build_reference_contract, find_key_frequencies


def make_contracts(model, p, heldout):
    ref = build_reference_contract(model, p, heldout)
    K = ref["components"]["elements"]

    empty = copy.deepcopy(ref)
    empty["components"]["elements"] = []

    whole = copy.deepcopy(ref)
    whole["components"]["elements"] = list(range(1, p // 2 + 1))

    wrong = copy.deepcopy(ref)
    wrong["components"]["elements"] = [(k + 3) % (p // 2) + 1 for k in K]

    blind = copy.deepcopy(ref)  # keeps optimistic predictions but with a wrong component set
    blind["components"]["elements"] = [1, 2, 3]

    return {"reference": ref, "empty": empty, "whole-model": whole,
            "off-by-a-few": wrong, "blind-guess": blind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    model = load_model(args.ckpt)
    p = model.cfg.p
    heldout = _heldout(p)

    print(f"true key frequencies: {find_key_frequencies(model, p)}\n")
    print(f"{'contract':<14}{'|components|':<14}{'score':<8}")
    print("-" * 36)
    for name, claim in make_contracts(model, p, heldout).items():
        score, _ = grade(model, claim, p, heldout)
        print(f"{name:<14}{len(claim['components']['elements']):<14}{score:<8.3f}")


if __name__ == "__main__":
    main()
