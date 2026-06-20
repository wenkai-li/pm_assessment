"""The generic, domain-agnostic prediction-contract grader. It depends only on the Domain interface,
so the same scoring runs for every domain; only domain.execute changes.
"""


def prefix_len(flags):
    """Length of the longest leading run of truthy flags (the depth-gated prefix)."""
    n = 0
    for f in flags:
        if f:
            n += 1
        else:
            break
    return n


def grade_ladder(domain, instance, contract, heldout, probes=0):
    """Grade a committed contract along a domain's dependency-ordered derivation ladder and return a
    result dict for the depth-calibrated judge (judge.score_episode). Shared by every domain that
    defines DERIVATION_LADDER, so the same depth-vs-premature-commitment reward applies everywhere:
    commit_depth is how deep the agent claimed (leading rungs supplied), correct_depth is how deep its
    claims actually hold on held-out conditions, and solved means the full derivation is verified."""
    ladder = domain.DERIVATION_LADDER
    preds = {p["id"]: p for p in contract["predictions"]}
    commit_depth = prefix_len(rung in preds for rung in ladder)
    pass_flags = []
    for rung in ladder[:commit_depth]:
        _, passed = domain.execute(preds[rung], instance, contract["components"], heldout)
        pass_flags.append(passed)
    correct_depth = prefix_len(pass_flags)
    required = len(ladder)
    return {"solved": correct_depth == required, "correct_depth": correct_depth,
            "commit_depth": commit_depth, "required": required, "probes": probes}


def grade(domain, instance, contract, heldout):
    structure = contract["components"]
    checks, details = {}, {}
    for pred in contract["predictions"]:
        actual, passed = domain.execute(pred, instance, structure, heldout)
        checks[pred["id"]] = passed
        details[pred["id"]] = actual

    gate = checks.get("necessity", False) and checks.get("sufficiency", False)
    mean_pass = sum(checks.values()) / max(len(checks), 1)
    size = domain.structure_size(structure)
    min_factor = max(0.0, 1.0 - size / max(domain.max_structure(instance), 1))
    score = mean_pass * min_factor if gate else 0.2 * mean_pass
    return score, {"checks": checks, "details": details, "gate": gate, "min_factor": min_factor}
