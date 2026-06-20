"""Run the exploration diagnostic over the depth-controlled task and print the table that separates
premature commitment from a capability ceiling.

  python3 diagnose.py --instances 200 --depth 8 --branching 4 --k 8

The headline: a policy that fails single-shot but passes pass@k and solves under forced continuation,
while committing shallower than the required depth, is diagnosed as PREMATURE COMMITMENT, not a
capability ceiling. A policy that fails every condition is diagnosed as a capability ceiling.
"""
import argparse

from judge import score_episode
from latent_chain import LatentChainTask
from protocol import (forced_continuation_solves, pass_at_k, run_episode,
                      solves_with_partial_hint)
from solvers import CapabilityLimited, Premature, Thorough

POLICIES = {"thorough": Thorough, "premature": Premature, "capability_limited": CapabilityLimited}


def verdict(realized, passk, forced, hint, commit_frac):
    ceiling = max(passk, forced, hint)
    if realized >= 0.9:
        return "solves single-shot"
    if ceiling - realized >= 0.3:
        if forced - realized >= 0.3 and commit_frac < 0.9:
            return "PREMATURE COMMITMENT (capability present; explores too shallow)"
        return "exploration-limited (capability present)"
    return "capability ceiling (fails even with help)"


def evaluate(PolicyCls, task, instances, k):
    realized = passk = forced = hint = 0
    reward = 0.0
    commit_fracs = []
    for inst in instances:
        r = run_episode(PolicyCls(), task, inst, attempt_seed=0)
        reward += score_episode(r).score
        realized += r["solved"]
        if not r["solved"]:
            commit_fracs.append(r["commit_depth"] / r["required"])
        passk += pass_at_k(PolicyCls(), task, inst, k)
        forced += forced_continuation_solves(PolicyCls(), task, inst)
        hint += solves_with_partial_hint(PolicyCls(), task, inst)
    n = len(instances)
    cf = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 1.0
    return {"reward": reward / n, "pass@1": realized / n, "pass@k": passk / n,
            "forced": forced / n, "hint<d*": hint / n, "commit/req(fail)": cf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--branching", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    task = LatentChainTask(depth=args.depth, branching=args.branching)
    instances = [task.build(seed) for seed in range(args.instances)]

    cols = ["reward", "pass@1", "pass@k", "forced", "hint<d*", "commit/req(fail)"]
    print(f"depth d* = {args.depth}, branching = {args.branching}, "
          f"instances = {args.instances}, k = {args.k}\n")
    header = f"{'policy':<20}" + "".join(f"{c:<18}" for c in cols) + "verdict"
    print(header)
    print("-" * len(header))
    for name, PolicyCls in POLICIES.items():
        m = evaluate(PolicyCls, task, instances, args.k)
        v = verdict(m["pass@1"], m["pass@k"], m["forced"], m["hint<d*"], m["commit/req(fail)"])
        row = f"{name:<20}" + "".join(f"{m[c]:<18.2f}" for c in cols) + v
        print(row)


if __name__ == "__main__":
    main()
