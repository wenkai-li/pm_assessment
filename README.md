# The Discovery — an RL environment for mechanistic-interpretability research

A proof of concept of an RL environment that asks an agent to reverse-engineer the internal
mechanism of a trained model and prove it understood the mechanism by predicting the effect of
causal interventions on held-out inputs. The grader is purely behavioral, with no language-model
judging.

This repository implements one mechanism template: a transformer trained on `(a + b) mod p`, whose
ground-truth algorithm (the Fourier "clock") is known, so verification is exact. The key frequencies
are randomized per training seed, so the answer has to be rediscovered for every model rather than
memorized.

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

The environment lives in `discovery-env/`. `model.py` and `train_modular_addition.py` build the model
zoo; `interventions.py` is the action space; `grader.py` scores a causal-claim contract;
`reference_solver.py` is the environment's own ground truth; `agent_runner.py` runs an LLM solver.
