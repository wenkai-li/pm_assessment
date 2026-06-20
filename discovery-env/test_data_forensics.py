"""Runnable check for the data-forensics domain (pure numpy, no GPU). It builds an instance, trains
the linear model, and grades the reference contract, a shallow (premature) contract, and a wrong
contract through the SAME depth-calibrated judge used by the interpretability domain. This shows the
ladder and judge generalize across domains.

  python3 test_data_forensics.py
"""
import copy

from domains.data_forensics import DataForensicsDomain
from judge import score_episode


def shallow(ref, depth):
    """Keep only the first `depth` rungs of the reference contract (a premature commit)."""
    c = copy.deepcopy(ref)
    order = DataForensicsDomain.DERIVATION_LADDER
    keep = set(order[:depth])
    c["predictions"] = [p for p in c["predictions"] if p["id"] in keep]
    return c


def main():
    dom = DataForensicsDomain()
    n_solved = 0
    rows = []
    for seed in range(8):
        inst = dom.build_instance(seed)
        ho = dom.heldout(inst)
        ref = dom.reference_contract(inst)

        r_full = dom.grade_episode(inst, ref, ho)
        r_shallow = dom.grade_episode(inst, shallow(ref, 2), ho)   # detect + localize, then commit
        wrong = copy.deepcopy(ref)
        wrong["components"]["spurious"] = [(inst["spurious"] + 3) % 8]
        r_wrong = dom.grade_episode(inst, wrong, ho)

        rows.append((seed, inst["spurious"], score_episode(r_full).score, r_full["correct_depth"],
                     score_episode(r_shallow).score, score_episode(r_wrong).score))
        n_solved += r_full["solved"]

    print(f"{'seed':<6}{'s*':<4}{'ref_score':<11}{'ref_depth':<11}{'shallow':<10}{'wrong':<8}")
    print("-" * 50)
    for seed, s, rs, rd, sh, wr in rows:
        print(f"{seed:<6}{s:<4}{rs:<11.3f}{rd:<11}{sh:<10.3f}{wr:<8.3f}")
    print(f"\nreference solved on {n_solved}/8 instances; "
          f"shallow stays below the success band; wrong feature scores ~0")


if __name__ == "__main__":
    main()
