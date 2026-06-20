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

Its central goal is to answer, for every failure, whether a model genuinely cannot reason the answer
out or whether it committed too early and did not explore deep enough. Each instance has a known
solution depth with depth-gated information, and an exploration diagnostic runs the same task
single-shot, under repeated sampling, with prefix hints, and under forced continuation to place each
failure on the line between a capability ceiling and a premature commitment.

This repository implements the model-internals domain as the worked instance: a transformer trained
on `(a + b) mod p`, whose ground-truth algorithm (the Fourier "clock") is known, so verification is
exact. The key frequencies are randomized per training seed, so the answer has to be rediscovered for
every model rather than memorized. Its prediction contract is a dependency-ordered derivation ladder
(necessity, sufficiency, specificity, algorithm, phase shift), so the same depth-calibrated judge that
scores the synthetic task scores this real one: naming the components without deriving how they compute
the answer is a shallow commit and scores below the success band. The other four domains are specified
as stubs that share the interface (`domains/`).

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

## The central measurement: capability vs premature commitment

`diagnose.py` runs the exploration diagnostic on a depth-controlled task and prints a table that
separates the two failure modes. It needs no GPU and runs in seconds:

```bash
cd discovery-env
python3 diagnose.py --instances 300 --depth 10 --branching 4 --k 8
```

A premature-commitment policy fails single-shot but passes pass@k and forced continuation while
committing at about half the required depth, and is labelled accordingly; a capability-limited policy
fails every condition. The same protocol wraps the real domains.

## Running it

See [RUNBOOK.md](RUNBOOK.md) for the full end-to-end procedure on a GPU cluster, including running a
real open-weights LLM (for example Llama) as the solver agent and measuring its failure modes.

```bash
cd discovery-env
pip install -r requirements.txt
python3 diagnose.py --instances 300 --depth 10           # exploration diagnostic (no GPU)
python3 train_modular_addition.py --seeds 0 1 2          # build the model zoo (grokking)
python3 calibration.py --ckpt checkpoints/seed_0.pt      # print the calibration table
python3 agent_runner.py --ckpt checkpoints/seed_0.pt     # run an LLM solver (needs a served model)
```

## Layout

The environment lives in `discovery-env/`. `domains/base.py` is the `Domain` interface and
`contract.py` is the domain-agnostic grader; `benchmark.py` runs it over any registered domain
(`python3 benchmark.py --list`). For the model-internals domain, `model.py` and
`train_modular_addition.py` build the model zoo, `interventions.py` is the action space, `grader.py`
and `reference_solver.py` implement the verifier and the ground truth, and `agent_runner.py` runs an
LLM solver. The other four domains are stubs under `domains/`. The per-episode reward is the
depth-calibrated judge in `judge.py` (success override, a success band so partial progress cannot
masquerade as success, and a probe budget that blocks brute-forcing the depth gate). The exploration
diagnostic lives in `latent_chain.py` (the depth-controlled task), `protocol.py` (the conditions),
`solvers.py` (scripted policies of known behavior), and `diagnose.py` (the table, which reports the
judge reward alongside the protocol signals).
