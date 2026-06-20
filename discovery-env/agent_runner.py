"""Run a real LLM (e.g. Llama) as the solver agent for The Discovery environment.

This is a code-agent harness, which matches the assessment framing (the base tool is a CLI that
reads/writes/runs files). The LLM is shown the task, given a restricted Python sandbox whose
namespace exposes the model and the intervention helpers, and asked to investigate and then submit
a causal-claim contract. The harness executes the LLM's code, returns stdout as the observation,
and finally grades the submitted contract with the same grader used for calibration.

SKELETON: adapt the backend to your serving setup. The default talks to an OpenAI-compatible
endpoint, which is what `vllm serve meta-llama/Llama-3.1-8B-Instruct` exposes.

SECURITY: this executes model-generated code. Run it inside the environment VM / a container, which
is what the RL environment assumes anyway. Do not run untrusted code on a shared host.
"""
import argparse
import contextlib
import io
import json
import re

import numpy as np
import requests
import torch

from fourier import make_fourier_basis
from grader import _heldout, grade
from interventions import ablate, accuracy, all_pairs, make_batch, model_logits, phase_shift, run
from io_utils import load_model

SYSTEM_PROMPT = """You are reverse-engineering the internal mechanism of a trained transformer that
computes (a + b) mod p for inputs a, b in {0,...,p-1}. You work by writing Python in a sandbox and
reading the output, then you submit a causal-claim contract that the grader checks on held-out
inputs.

The sandbox namespace already contains:
  model            the trained transformer (frozen)
  p                the modulus
  inspect_embedding()        -> the (p, d) number-token embedding matrix as a numpy array
  acc(pairs=None)            -> accuracy on given (a,b) pairs, or on a default sample
  ablate_freqs(freqs, mode)  -> accuracy after removing ('remove') or keeping-only ('keep') freqs
  phase_test(freq, shift=1)  -> fraction of inputs whose answer shifts by `shift` under a phase
                                rotation at `freq` (the directional test)
  fourier_basis()            -> the (p, p) orthonormal Fourier basis over token indices
  np, torch

Respond with EXACTLY ONE fenced block per turn, either:
  ```python
  # code to run; use print(...) to observe
  ```
or, when you are confident, a final submission:
  ```json
  {"components": {"basis": "fourier", "elements": [...]},
   "predictions": [
     {"id": "necessity", "predicted": <x>, "tol": <t>},
     {"id": "sufficiency", "predicted": <x>, "tol": <t>},
     {"id": "specificity", "probe": [...], "predicted": [...], "tol": <t>},
     {"id": "algorithm", "predicted": <corr>, "tol": <t>},
     {"id": "phase_shift", "freq": <k>, "shift": 1, "predicted": <rate>, "tol": <t>}]}
  ```
You are graded only on whether the predicted effects hold on inputs you have not seen. There is no
single correct component set. Commit to specific numbers; do not hedge."""


def llm_chat(messages, base_url, model_name, api_key="EMPTY", temperature=0.7, max_tokens=1200):
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_name, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def build_sandbox(model, p):
    def inspect_embedding():
        return model.W_E.data[:p].clone().numpy()

    def acc(pairs=None):
        if pairs is None:
            pairs = all_pairs(p)[:: max(1, p * p // 400)]
        tok, tgt = make_batch(pairs, p)
        return accuracy(run(model, tok), tgt)

    def ablate_freqs(freqs, mode="remove"):
        tok, tgt = make_batch(all_pairs(p)[:: max(1, p * p // 400)], p)
        return ablate(model, tok, tgt, list(freqs), p, mode=mode)

    def phase_test(freq, shift=1):
        tok, _ = make_batch(all_pairs(p)[:: max(1, p * p // 400)], p)
        logits, predicted = phase_shift(model, tok, p, freq, shift)
        return (logits.argmax(-1) == predicted).float().mean().item()

    return {
        "model": model, "p": p, "np": np, "torch": torch,
        "inspect_embedding": inspect_embedding, "acc": acc,
        "ablate_freqs": ablate_freqs, "phase_test": phase_test,
        "fourier_basis": lambda: make_fourier_basis(p),
    }


def run_code(code, ns):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        out = buf.getvalue()
        return (out[-3500:] or "(no output; remember to print)")
    except Exception as e:  # surface the error to the agent so it can recover
        return f"ERROR: {type(e).__name__}: {e}"


def extract_block(text, lang):
    m = re.search(rf"```{lang}\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--log", default="run_trace.json")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    p = model.cfg.p
    ns = build_sandbox(model, p)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.replace("p ", f"p={p} ")}]
    messages.append({"role": "user", "content": "Begin. The modulus is p = %d." % p})

    trace = {"tool_calls": 0, "patch_calls": 0, "steps": []}
    for step in range(args.max_steps):
        reply = llm_chat(messages, args.base_url, args.model_name)
        messages.append({"role": "assistant", "content": reply})

        claim_src = extract_block(reply, "json")
        if claim_src:
            claim = json.loads(claim_src)
            score, info = grade(model, claim, p, _heldout(p), verbose=True)
            trace["final_score"] = score
            trace["final_checks"] = info.get("checks")
            trace["steps"].append({"step": step, "type": "submit", "score": score})
            print(f"\n=== FINAL SCORE {score:.3f} | steps used {step} | "
                  f"patch/ablate calls {trace['patch_calls']} ===")
            break

        code = extract_block(reply, "python")
        if code is None:
            messages.append({"role": "user", "content": "Reply with exactly one ```python``` or ```json``` block."})
            continue
        trace["tool_calls"] += 1
        if "ablate_freqs" in code or "phase_test" in code:
            trace["patch_calls"] += 1  # proxy for genuine causal-intervention use
        obs = run_code(code, ns)
        trace["steps"].append({"step": step, "type": "code", "code": code, "obs": obs[:1000]})
        messages.append({"role": "user", "content": f"Output:\n{obs}"})
    else:
        print("\n=== NO SUBMISSION within max steps (a 'does not commit' failure) ===")
        trace["final_score"] = 0.0

    with open(args.log, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"trace written to {args.log}")


if __name__ == "__main__":
    main()
