# discovery-env (proof of concept)

A minimal, runnable slice of **The Discovery** — an RL environment that asks an agent to
reverse-engineer the internal mechanism of a trained model and prove it understood the mechanism
by predicting the effect of causal interventions on held-out inputs.

This POC implements ONE mechanism template: a transformer trained on modular addition
`(a + b) mod P`, whose ground-truth algorithm (a Fourier "clock") is known, so verification is
exact. The point of the POC is the **calibration table**: it shows empirically that the grader
gives a high score to the true mechanism and a near-zero score to reward hacks — i.e. no false
positives and no false negatives, demonstrated rather than asserted.

## Layout
- `model.py` — a small 1-layer transformer (Config + Transformer), forward accepts `input_embeds`.
- `train_modular_addition.py` — trains the model to grokking for several seeds; saves checkpoints.
- `fourier.py` — Fourier basis over token indices; filter / phase-shift the embedding matrix.
- `interventions.py` — the action space: run, accuracy, ablate (remove / keep-only), phase-shift.
- `reference_solver.py` — the env's own ground-truth: find the key frequencies, build the
  reference causal-claim contract.
- `grader.py` — reads a `claim.json` contract and scores it by running each intervention on
  held-out inputs (necessity, sufficiency, specificity, algorithm, phase-shift; minimality bonus).
- `calibration.py` — builds the reference contract and several hack / wrong contracts and prints
  the no-FP/no-FN table.
- `example_claim.json` — the schema a solver agent must produce.

## Run
```
pip install -r requirements.txt
python train_modular_addition.py --seeds 0 1 2      # trains + saves checkpoints/seed_*.pt
python calibration.py --ckpt checkpoints/seed_0.pt  # prints the calibration table
python grader.py --ckpt checkpoints/seed_0.pt --claim example_claim.json
```

## What the calibration table demonstrates
| contract | what it is | expected score |
|---|---|---|
| reference | true key frequencies + true algorithm | ~1.0 (no false negative) |
| empty | no components claimed | ~0.0 |
| whole-model | claim every frequency matters | low (minimality + specificity penalty) |
| off-by-a-few | plausible but wrong frequency set | ~0.0 (no false positive) |
| blind-guess | predicts all checks pass without understanding | ~0.0 |

This POC is a skeleton meant to be read and run, not a production environment. The same machinery
generalizes to other mechanism templates (induction heads, IOI routing, SAE features).
