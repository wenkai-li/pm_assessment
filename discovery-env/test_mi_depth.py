"""Validate the MI derivation-ladder grading and its wiring to the depth-calibrated judge, without a
trained model. A mock domain returns canned pass/fail per rung so the depth-prefix logic and the judge
score can be checked directly.

  python3 test_mi_depth.py
"""
from domains.modular_addition import ModularAdditionDomain
from judge import score_episode

LADDER = ModularAdditionDomain.DERIVATION_LADDER


class MockMI(ModularAdditionDomain):
    """Pass the first `passes` rungs, fail the rest, regardless of the predictions' numbers."""
    def __init__(self, passes):
        super().__init__(ckpt=None)
        self._passes = passes

    def execute(self, pred, instance, structure, heldout):
        rung_index = LADDER.index(pred["id"])
        return (None, rung_index < self._passes)


def contract_with_rungs(n):
    return {"components": {"basis": "fourier", "elements": [5, 17]},
            "predictions": [{"id": LADDER[i]} for i in range(n)]}


def check(name, claimed_rungs, passing_rungs, expect_correct, expect_solved, score_pred):
    dom = MockMI(passes=passing_rungs)
    res = dom.grade_episode(instance={}, contract=contract_with_rungs(claimed_rungs), heldout=None)
    j = score_episode(res)
    assert res["commit_depth"] == claimed_rungs, (name, res)
    assert res["correct_depth"] == expect_correct, (name, res)
    assert res["solved"] == expect_solved, (name, res)
    assert score_pred(j.score), (name, j.score)
    print(f"{name:<42} commit={res['commit_depth']} correct={res['correct_depth']} "
          f"solved={res['solved']} score={j.score:.3f}")


if __name__ == "__main__":
    # Full mechanism: all 5 rungs claimed and all pass -> solved, score 1.0.
    check("full mechanism", 5, 5, 5, True, lambda s: s == 1.0)
    # Shallow/premature: names components and shows necessity+sufficiency, then commits (no algorithm).
    check("premature shallow (commit at 2)", 2, 5, 2, False, lambda s: s < 0.5)
    # Deep claim but the algorithm rung is wrong -> verified depth stops at 3.
    check("claims 5, algorithm wrong", 5, 3, 3, False, lambda s: s < 0.5)
    # Over-claim with nothing right -> correct depth 0, near-zero score.
    check("claims 5, nothing holds", 5, 0, 0, False, lambda s: s == 0.0)
    print("\nMI DEPTH TESTS PASSED")
