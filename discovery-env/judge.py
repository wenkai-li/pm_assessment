"""The per-episode judge: a depth-calibrated, depth-gated score.

This is the reward the assessment asks for. It scores ONE attempt, grounded in a known required depth
d* with depth-gated verification, so it directly rewards descending deep enough before committing
rather than committing prematurely. The design was stress-tested with an independent review (Codex,
gpt-5.5); the resulting choices are noted inline.

Scoring (continuous, [0, 1]):
  - Success override: a fully correct terminal answer scores exactly 1.0, regardless of how the agent
    got there. This prevents reward denial for a solver that reasons ahead without probing every gate.
  - Partial credit below a success band: a non-solved attempt scores at most tau (< 0.5), so partial
    exploration can never masquerade as success. Partial credit is tau * (k / d*) ** gamma, where k is
    the longest correct prefix of the committed answer, checked against ground truth at grading time.
    Using genuinely-correct depth k (not self-reported depth) means over-claiming cannot inflate the
    score, and a large branching factor makes lucky guesses past the verified frontier negligible.
  - Hard probe budget (optional): if the agent exceeds q_max oracle probes, the episode scores 0. This
    is the brute-force defense. It is off by default for the synthetic instrument, whose only way to
    find a key is to try, and is set in the real ML-research domains, where producing a candidate step
    requires reasoning or running an experiment rather than enumerating options. Within budget a
    correct solver is never penalized for probe count, which keeps the judge free of false negatives.

Commitment depth, over-claim, and probe counts are returned as DIAGNOSTICS. The exploration protocol
(pass@k, forced continuation, prefix-hint sweep) consumes them to separate premature commitment from a
capability ceiling; they are not folded into the reward, because turning them into penalties would
deny reward to legitimate solvers.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class JudgeResult:
    score: float           # continuous reward in [0, 1]
    solved: bool           # terminal answer fully correct
    correct_depth: int     # longest correct prefix of the committed answer (k)
    verified_depth: int    # depth the agent actually passed the gate to (v)
    commit_depth: int      # depth verified at the moment of commit
    required_depth: int    # d*
    probes: int            # oracle queries used
    overclaim: int         # correct-committed depth beyond gate-verified depth (luck, not earning)
    budget_exceeded: bool  # probes > q_max


def score_episode(result: dict, tau: float = 0.49, gamma: float = 1.0,
                  q_max: Optional[int] = None) -> JudgeResult:
    """Score one episode from a protocol.run_episode result dict."""
    d_star = result["required"]
    k = result.get("correct_depth", result["commit_depth"])
    v = result["commit_depth"]
    probes = result.get("probes", 0)
    solved = bool(result.get("solved", False))

    budget_exceeded = q_max is not None and probes > q_max
    if budget_exceeded:
        score = 0.0
    elif solved:
        score = 1.0                                   # success override (no false negative)
    elif d_star <= 0:
        score = 0.0
    else:
        depth_frac = max(0.0, min(1.0, k / d_star))
        score = tau * depth_frac ** gamma             # partial stays below the success band

    return JudgeResult(
        score=score, solved=solved, correct_depth=k, verified_depth=v, commit_depth=v,
        required_depth=d_star, probes=probes, overclaim=max(0, k - v),
        budget_exceeded=budget_exceeded,
    )
