"""The generic, domain-agnostic prediction-contract grader. It depends only on the Domain interface,
so the same scoring runs for every domain; only domain.execute changes.
"""


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
