"""Run an external coding agent (Claude Code or Codex) as a solver on the modular-addition internals
environment, then grade its committed claim with the depth-calibrated judge.

This is the baseline experiment: it measures whether top agents can produce a verified mechanistic
account, and how deep into the derivation they get before they commit. It builds an isolated workspace
that contains the trained model and the safe investigation tools, but NOT the grader, the reference
solution, or the judge, so the agent cannot read the answer or the scoring code.

  python3 baseline/run_baseline.py --ckpt checkpoints/seed_0.pt --agent codex
  python3 baseline/run_baseline.py --ckpt checkpoints/seed_0.pt --agent claude

Requires the corresponding CLI on PATH (`codex` or `claude`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.dirname(HERE)
SAFE_MODULES = ["model.py", "fourier.py", "interventions.py", "io_utils.py"]

PROMPT = """You are given a trained transformer in ./model.pt that computes (a + b) mod 113 for
integer inputs a, b in {0,...,112}. It is close to 100 percent accurate. Reverse-engineer the internal
mechanism it uses, then demonstrate understanding by predicting the outcome of causal interventions.

Use the helpers in tools.py (import tools). Available:
  tools.load()                         -> the model
  tools.accuracy(pairs=None)           -> accuracy on given (a,b) pairs or a default sample
  tools.ablate_freqs(freqs, mode)      -> accuracy after removing ('remove') or keeping only ('keep')
                                          the listed Fourier frequencies of the input embedding
  tools.phase_test(freq, shift=1)      -> fraction of inputs whose answer shifts by `shift` under a
                                          phase rotation at `freq`
  tools.fourier_basis()                -> the (113,113) Fourier basis over token indices
  tools.inspect_embedding()            -> the (113, d) number-token embedding matrix
numpy (including numpy.fft) and torch are installed. Write and run your own scripts to investigate.

When done, write claim.json with this schema (you are graded only on whether the predicted effects
hold on inputs you have not seen; there is no single correct component set):

{
  "components": {"basis": "fourier", "elements": [<key frequencies, ints in 1..56>]},
  "predictions": [
    {"id": "necessity",   "predicted": <acc after removing the key freqs>, "tol": 0.05},
    {"id": "sufficiency", "predicted": <acc keeping only the key freqs>,   "tol": 0.05},
    {"id": "specificity", "probe": [<freqs>], "predicted": [<acc removing each one>], "tol": 0.06},
    {"id": "algorithm",   "predicted": <corr of cos(2*pi*k*(a+b-c)/113) sum with logits>, "tol": 0.05},
    {"id": "phase_shift", "freq": <a key freq>, "shift": 1, "predicted": <shift rate>, "tol": 0.1}
  ]
}
Commit specific numbers. Do not hedge. Stop and write claim.json when you are confident."""


def build_workspace(ckpt):
    ws = tempfile.mkdtemp(prefix="mi_baseline_")
    for m in SAFE_MODULES:
        shutil.copy(os.path.join(ENV, m), os.path.join(ws, m))
    shutil.copy(ckpt, os.path.join(ws, "model.pt"))
    with open(os.path.join(ws, "tools.py"), "w") as f:
        f.write(TOOLS_PY)
    with open(os.path.join(ws, "PROMPT.txt"), "w") as f:
        f.write(PROMPT)
    return ws


TOOLS_PY = '''"""Investigation helpers for the modular-addition internals task. The model and the safe
interventions only; no grader, no reference answer."""
import numpy as np, torch
from io_utils import load_model
from interventions import make_batch, all_pairs, accuracy, run, ablate, phase_shift
from fourier import make_fourier_basis

P = 113
_model = None

def load():
    global _model
    if _model is None:
        _model = load_model("model.pt")
    return _model

def _sample(n=400):
    pairs = all_pairs(P)
    return pairs[:: max(1, len(pairs)//n)]

def accuracy(pairs=None):
    m = load(); pairs = pairs or _sample()
    tok, tgt = make_batch(pairs, P)
    from interventions import accuracy as acc
    return acc(run(m, tok), tgt)

def ablate_freqs(freqs, mode="remove"):
    m = load(); tok, tgt = make_batch(_sample(), P)
    return ablate(m, tok, tgt, list(freqs), P, mode=mode)

def phase_test(freq, shift=1):
    m = load(); tok, _ = make_batch(_sample(), P)
    logits, predicted = phase_shift(m, tok, P, freq, shift)
    return (logits.argmax(-1) == predicted).float().mean().item()

def fourier_basis():
    return make_fourier_basis(P)

def inspect_embedding():
    return load().W_E.data[:P].clone().numpy()
'''


def run_agent(agent, ws):
    prompt_path = os.path.join(ws, "PROMPT.txt")
    with open(prompt_path) as f:
        prompt = f.read()
    if agent == "codex":
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
               "-C", ws, prompt]
    elif agent == "claude":
        cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
    else:
        raise SystemExit(f"unknown agent {agent}")
    print(f"running {agent} in {ws} ...", flush=True)
    subprocess.run(cmd, cwd=ws, timeout=1800)


def grade(ws, ckpt):
    claim_path = os.path.join(ws, "claim.json")
    if not os.path.exists(claim_path):
        return {"score": 0.0, "note": "no claim.json produced (did not commit)"}
    sys.path.insert(0, ENV)
    from domains.modular_addition import ModularAdditionDomain
    from judge import score_episode
    with open(claim_path) as f:
        claim = json.load(f)
    dom = ModularAdditionDomain(ckpt=ckpt)
    inst = dom.build_instance(seed=0)
    res = dom.grade_episode(inst, claim, dom.heldout(inst))
    j = score_episode(res)
    return {"score": j.score, "solved": j.solved, "correct_depth": j.correct_depth,
            "required_depth": j.required_depth, "claim": claim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--agent", choices=["codex", "claude"], required=True)
    ap.add_argument("--keep", action="store_true", help="keep the workspace for inspection")
    args = ap.parse_args()
    ckpt = os.path.abspath(args.ckpt)
    ws = build_workspace(ckpt)
    try:
        run_agent(args.agent, ws)
        result = grade(ws, ckpt)
        print("\n=== BASELINE RESULT ===")
        print(json.dumps({k: v for k, v in result.items() if k != "claim"}, indent=2))
    finally:
        if not args.keep:
            shutil.rmtree(ws, ignore_errors=True)
        else:
            print(f"workspace kept at {ws}")


if __name__ == "__main__":
    main()
