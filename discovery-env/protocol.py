"""The exploration diagnostic protocol.

It runs the same verifiable task under several conditions to separate a capability ceiling (the
policy cannot solve it even with help) from premature commitment (the policy can solve it but
commits before descending to the required depth). Every signal here is verifiable; nothing is judged
by a language model.

Conditions:
  pass@1               single-shot solve rate (realized performance).
  pass@k               solve within k samples (capability under sampling).
  prefix-hint sweep    give the first j correct steps for j < d*; can it finish? (localizes the depth
                       at which autonomous exploration, not reasoning, breaks).
  forced continuation  reject early commits until depth d* is reached; does it then solve?
  commitment depth     the depth verified at the moment of commit, relative to d* (premature if < d*).
"""


def run_episode(policy, task, inst, hint=0, force_full=False, attempt_seed=0, step_budget=None):
    policy.reset(inst, hint, attempt_seed)
    D = task.required_depth(inst)
    B = inst["branching"]
    verified = hint
    probes = 0
    budget = step_budget if step_budget is not None else D * B * 6
    for _ in range(budget):
        kind, payload = policy.act(verified)
        if kind == "probe":
            prefix = list(payload)
            probes += 1
            ok = task.on_path(inst, prefix)
            if ok and len(prefix) == verified + 1:
                verified = len(prefix)
            policy.observe(prefix, ok)
        else:  # submit
            if force_full and verified < D:
                policy.observe_rejected()
                continue
            return {"solved": task.verify(inst, payload), "commit_depth": verified,
                    "required": D, "probes": probes}
    return {"solved": False, "commit_depth": verified, "required": D, "probes": probes,
            "timeout": True}


def pass_at_k(policy, task, inst, k):
    for a in range(k):
        if run_episode(policy, task, inst, attempt_seed=a)["solved"]:
            return True
    return False


def solves_with_partial_hint(policy, task, inst):
    """True if the policy finishes from some hint depth j strictly below d* (capability via a hint
    that does not give away the answer)."""
    D = task.required_depth(inst)
    for j in range(0, D):  # j < D, so the last key is never handed over
        if run_episode(policy, task, inst, hint=j)["solved"]:
            return True
    return False


def forced_continuation_solves(policy, task, inst):
    return run_episode(policy, task, inst, force_full=True)["solved"]
