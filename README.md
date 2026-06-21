# The Discovery Benchmark — proof of concept

An RL environment for research discovery, where an agent investigates a system with a hidden,
randomized structure and proves it understood by predicting the effect of causal interventions on
held-out conditions. The judge is purely behavioral (no language-model judge), and a depth-calibrated
score rewards exploring deep enough before committing. Two domains are implemented (model internals,
data forensics); three more are specified stubs. The design rationale lives in the accompanying
submission document; this repo is the runnable code.

## Setup

```bash
cd discovery-env
pip install -r requirements.txt          # torch, numpy
```

## Run

No GPU needed for these:

```bash
# Exploration diagnostic: separates premature commitment from a capability ceiling
python3 diagnose.py --instances 300 --depth 10 --branching 4 --k 8

# Second domain (data forensics) calibration table
python3 benchmark.py --domain data_forensics

# Tests
python3 test_mi_depth.py
python3 test_data_forensics.py
```

Model-internals domain (trains a small transformer to grokking, then grades):

```bash
python3 train_modular_addition.py --seeds 0 1 2     # build the model zoo (grokking, CPU ~10-15 min/seed)
python3 calibration.py --ckpt checkpoints/seed_0.pt # no-false-positive / no-false-negative table
python3 grader.py --ckpt checkpoints/seed_0.pt --claim example_claim.json
```

Baseline with an external coding agent (needs `codex` or `claude` on PATH and a trained checkpoint):

```bash
python3 baseline/run_baseline.py --ckpt checkpoints/seed_0.pt --agent codex
python3 baseline/run_baseline.py --ckpt checkpoints/seed_0.pt --agent claude
```

See [RUNBOOK.md](RUNBOOK.md) for the full end-to-end procedure and expected outputs.

## Layout

```
discovery-env/
  judge.py, contract.py            depth-calibrated judge + shared derivation-ladder grader
  domains/                         Domain interface + modular_addition, data_forensics (impl), 3 stubs
  model.py, fourier.py, interventions.py, grader.py, reference_solver.py, calibration.py
                                   model-internals (modular addition) domain
  latent_chain.py, protocol.py, solvers.py, diagnose.py
                                   exploration diagnostic (capability vs premature commitment)
  baseline/run_baseline.py         run Claude Code / Codex as solvers; results in baseline/results/
  test_mi_depth.py, test_data_forensics.py
```
