# The Discovery Benchmark — an RL environment for research discovery

A benchmark, in the same sense as SWE-Bench: one uniform verifier run over many diverse instances.
Every instance presents a system with a hidden, randomly placed structure, and the agent must
investigate it and prove understanding by submitting a prediction contract, a set of causal
interventions plus quantitative predictions that the grader runs on held-out conditions. The grader
is purely behavioral, with no language-model judging.

The benchmark spans five domains of machine learning research: model internals, training dynamics,
data forensics, inference systems, and learned-policy behavior. They share one verifier through a
small `Domain` interface (`domains/base.py`), so adding a domain is adding a backend, not a new
grader.

This repository implements the model-internals domain as the worked instance: a transformer trained
on `(a + b) mod p`, whose ground-truth algorithm (the Fourier "clock") is known, so verification is
exact. The key frequencies are randomized per training seed, so the answer has to be rediscovered for
every model rather than memorized. The other four domains are specified as stubs that share the
interface (`domains/`).

## Why it is interesting

The environment scores the part of research that outcome-only benchmarks leave out: the investigative
loop from internal signals to a verified causal account. It rewards genuine understanding while
staying fully verifiable, because it reuses interpretability's own standard for whether an
explanation is real, the causal intervention. A claim is accepted only if it is necessary,
sufficient, specific, reproduces the computation in closed form, and predicts the signed effect of a
controlled manipulation on inputs the agent has not seen.

## The headline result: a calibration table

`calibration.py` builds the true contract and several wrong or hacking contracts and grades them all.
Only the true contract scores well; empty, whole-model, off-by-a-few, and blind-guess contracts score
near zero. This demonstrates the no-false-positive and no-false-negative property as a measured
result rather than a claim.

## Running it

See [RUNBOOK.md](RUNBOOK.md) for the full end-to-end procedure on a GPU cluster, including running a
real open-weights LLM (for example Llama) as the solver agent and measuring its failure modes.

```bash
cd discovery-env
pip install -r requirements.txt
python3 train_modular_addition.py --seeds 0 1 2     # build the model zoo (grokking)
python3 calibration.py --ckpt checkpoints/seed_0.pt # print the calibration table
python3 agent_runner.py --ckpt checkpoints/seed_0.pt # run an LLM solver (needs a served model)
```

## Layout

The environment lives in `discovery-env/`. `domains/base.py` is the `Domain` interface and
`contract.py` is the domain-agnostic grader; `benchmark.py` runs it over any registered domain
(`python3 benchmark.py --list`). For the model-internals domain, `model.py` and
`train_modular_addition.py` build the model zoo, `interventions.py` is the action space, `grader.py`
and `reference_solver.py` implement the verifier and the ground truth, and `agent_runner.py` runs an
LLM solver. The other four domains are stubs under `domains/`.
