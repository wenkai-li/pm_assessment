"""The action space exposed to the agent, and used by the grader to execute a contract.

Everything is expressed as a forward pass with a modified embedding matrix, which keeps the POC
small while staying faithful to where the modular-addition mechanism lives.
"""
import math

import torch

from fourier import filter_embedding, phase_shift_embedding


def all_pairs(p: int):
    return [(a, b) for a in range(p) for b in range(p)]


def make_batch(pairs, p: int):
    """Tokens [a, b, '='] and targets (a+b) mod p."""
    tokens = torch.tensor([[a, b, p] for (a, b) in pairs])
    targets = torch.tensor([(a + b) % p for (a, b) in pairs])
    return tokens, targets


@torch.no_grad()
def run(model, tokens, W_E=None, pos0_W_E=None):
    """Forward pass. `W_E` overrides the embedding for all positions; `pos0_W_E` overrides only
    position 0 (the `a` input), used by the phase-shift test."""
    base = model.W_E if W_E is None else W_E
    embeds = base[tokens]
    if pos0_W_E is not None:
        embeds = embeds.clone()
        embeds[:, 0, :] = pos0_W_E[tokens[:, 0]]
    return model(input_embeds=embeds)


def accuracy(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


@torch.no_grad()
def ablate(model, tokens, targets, freqs, p, mode="remove"):
    """Accuracy after removing (mode='remove') or keeping only (mode='keep') the given frequencies."""
    if mode == "remove":
        W = filter_embedding(model.W_E.data, p, remove=freqs)
    else:
        W = filter_embedding(model.W_E.data, p, keep=freqs)
    return accuracy(run(model, tokens, W_E=W), targets)


@torch.no_grad()
def phase_shift(model, tokens, p, freq, shift=1):
    """Phase-shift position 0 by `shift` input-units at `freq`; return predicted answers and logits."""
    delta = 2 * math.pi * freq * shift / p
    W_shift = phase_shift_embedding(model.W_E.data, p, freq, delta)
    logits = run(model, tokens, pos0_W_E=W_shift)
    a, b = tokens[:, 0], tokens[:, 1]
    predicted = (a + b + shift) % p
    return logits, predicted


@torch.no_grad()
def model_logits(model, tokens):
    return run(model, tokens)
