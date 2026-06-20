# RUNBOOK — running The Discovery on a GPU cluster

This document is self-contained. It explains the goal, the design, and the exact steps to run the
experiment end to end on a cluster, including running a real open-weights LLM (for example Llama) as
the solver agent. Read it top to bottom; each phase builds on the previous one.

---

## 0. What this experiment is

The Discovery is a reinforcement-learning environment that asks an agent to do mechanistic
interpretability. The agent is given a trained model that produces a fixed behavior, and it must
recover the internal mechanism and prove it understood by predicting the effect of causal
interventions it designs. The grader runs those interventions on held-out inputs and checks the
predictions, with no language-model judging.

This repository implements one mechanism template: a small transformer trained on `(a + b) mod p`,
whose ground-truth algorithm is the Fourier "clock" (each input is represented by trigonometric
features at a few key frequencies, multiplied inside the network, and read out through a
trigonometric identity). The key frequencies depend on the training seed, so the answer cannot be
memorized and must be rediscovered for each model.

The experiment has two goals:
1. Verify the grader has no false positives and no false negatives, using a calibration table.
2. Run a real LLM as the solver agent and measure how it does. The motivation is to see whether the
   LLM shows the two failure modes that motivated this environment: it does not commit to a definite
   tested conclusion, and it underuses activation patching, staying on shallow signals.

## 1. What is in the repository

- `discovery-env/model.py` — the transformer (forward accepts injected embeddings).
- `discovery-env/train_modular_addition.py` — trains to grokking for one or more seeds; early-stops
  when test accuracy passes 0.995 and saves `checkpoints/seed_*.pt`.
- `discovery-env/fourier.py` — Fourier basis over token indices; frequency filtering; phase shift.
- `discovery-env/interventions.py` — the action space: run, accuracy, ablate (remove / keep-only),
  phase shift.
- `discovery-env/reference_solver.py` — the environment's own ground truth: find the key
  frequencies and build the reference contract used for calibration.
- `discovery-env/grader.py` — scores a causal-claim contract (necessity, sufficiency, specificity,
  algorithm, phase shift; minimality bonus). Rejects malformed claims.
- `discovery-env/calibration.py` — builds the reference and several wrong / hacking contracts and
  prints the no-false-positive / no-false-negative table.
- `discovery-env/agent_runner.py` — the harness that runs a real LLM as the solver agent.
- `discovery-env/example_claim.json` — the contract schema.

## 2. Cluster setup

Use one GPU. CPU works for everything except it makes grokking slow and LLM inference impractical.

```bash
git clone https://github.com/wenkai-li/pm_assessment.git
cd pm_assessment/discovery-env
conda create -n discovery python=3.11 -y && conda activate discovery
pip install -r requirements.txt          # torch, numpy
```

For Phase 3 (the LLM solver) also install a serving stack:
```bash
pip install vllm requests                # serves Llama with an OpenAI-compatible API
```

## 3. Phase 0 — smoke test (1 minute)

Confirm the plumbing before training anything.
```bash
python3 -c "
import torch, json
from model import Config, Transformer
from interventions import make_batch, all_pairs, accuracy
from grader import grade, _heldout
m = Transformer(Config())
claim = json.load(open('example_claim.json'))
s, info = grade(m, claim, m.cfg.p, _heldout(m.cfg.p, n=80))
print('untrained score', round(s,3), 'gate', info['gate'])   # expect a low score, gate False
"
```
An untrained model should score low. A malformed claim (frequencies above p/2) should score 0.

## 4. Phase 1 — build the environment (train the model zoo)

```bash
python3 train_modular_addition.py --seeds 0 1 2 --steps 40000
```
Expected behavior is the grokking curve: the training loss falls to near zero within a few hundred
steps while test accuracy stays near chance (about 1/113) for a long time, then test accuracy jumps
to near 1.0 once the model generalizes. The script early-stops and saves a checkpoint when test
accuracy passes 0.995.

If a seed does not grok within the step budget, increase weight decay or the budget, or change the
training fraction:
- weight decay `wd` (default 1.0) is the main driver of grokking; try 0.5 to 2.0.
- `train_frac` (default 0.3) can go up to 0.5 to grok sooner.
- try several `--seeds`; some seeds grok faster than others.
Edit these in `train_modular_addition.py` (`train_one`).

## 5. Phase 2 — verify the judge (the no-FP / no-FN evidence)

```bash
python3 calibration.py --ckpt checkpoints/seed_0.pt
```
Expected table:

| contract | meaning | expected score |
|---|---|---|
| reference | true key frequencies + true algorithm | close to 1.0 (no false negative) |
| empty | no components claimed | about 0 |
| whole-model | every frequency claimed | low (minimality and specificity penalty) |
| off-by-a-few | plausible but wrong frequency set | about 0 (no false positive) |
| blind-guess | optimistic predictions, wrong components | about 0 |

This table is the core artifact. It shows that the only contract that scores well is the true one,
which is the no-false-positive and no-false-negative property demonstrated rather than asserted.
Save the printed table; it is what the assessment links to.

You can also grade a hand-written contract:
```bash
python3 grader.py --ckpt checkpoints/seed_0.pt --claim example_claim.json
```

## 6. Phase 3 — run a real LLM as the solver agent

This is the experiment you care about. Serve an open-weights model and let it solve the task through
the code-agent harness.

Serve Llama (one terminal):
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Run the solver (another terminal):
```bash
python3 agent_runner.py \
  --ckpt checkpoints/seed_0.pt \
  --base-url http://localhost:8000/v1 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --max-steps 20 \
  --log run_trace.json
```
The harness shows the model the task, gives it a Python sandbox whose namespace exposes the trained
model and the intervention helpers (`inspect_embedding`, `acc`, `ablate_freqs`, `phase_test`,
`fourier_basis`), executes the code it writes, returns the output, and finally grades the contract
it submits. The harness writes `run_trace.json`.

What to read from the trace, tied to the motivation:
- `final_score` — did the model produce a verified account at all.
- whether it submitted before `--max-steps` — not submitting is the "does not commit" failure.
- `patch_calls` versus `tool_calls` — how often it actually ran ablation or phase-shift
  interventions rather than only reading activations. A low ratio is the "underuses activation
  patching" failure.

Run it across several checkpoints and several samples to get a rate rather than a single point:
```bash
for s in 0 1 2; do
  for r in 1 2 3; do
    python3 agent_runner.py --ckpt checkpoints/seed_$s.pt --log trace_${s}_${r}.json
  done
done
```
Aggregate `final_score`, submission rate, and the patch-call ratio across runs.

Try stronger models the same way (for example Llama-3.1-70B-Instruct, or any OpenAI-compatible
endpoint) to see how the failure modes change with capability.

## 7. Phase 4 — what the result means

A low pass rate with frequent non-submission and few patch calls is direct evidence for the two
failure modes the environment was built to expose: the model does not commit to a tested conclusion,
and it underuses causal intervention. That evidence is the point. It says that this kind of internal
investigation is a real gap in current agents, and that the environment gives a verifiable,
trainable signal for closing it. The next step beyond this POC is to use the environment for
reinforcement learning, using the snapshot-and-restore interface to assign per-step credit, which is
described in the design write-up.

## 8. Troubleshooting

- No grokking: raise `wd`, raise `train_frac`, raise `--steps`, or try other seeds.
- `calibration.py` reference score not near 1.0: the model has not fully grokked, or
  `find_key_frequencies` threshold in `reference_solver.py` needs adjusting for this model; lower
  `rel_threshold` to include more frequencies, then re-check sufficiency.
- LLM harness parsing issues: some models wrap code differently. The harness extracts the first
  fenced ```python``` or ```json``` block; adjust `extract_block` if your model uses another format.
- Executing model code: run inside the environment VM or a container. The harness runs
  model-generated Python by design, which mirrors the CLI tool the real environment provides.
