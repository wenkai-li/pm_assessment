"""Fourier basis over token indices, and operations that filter or phase-shift the embedding
matrix along the token axis. The modular-addition mechanism lives in how the embedding represents
each input value as a function of its index, so this is where the interventions act.
"""
import math

import torch


def make_fourier_basis(p: int) -> torch.Tensor:
    """Return a (p, p) orthonormal basis over token indices 0..p-1.

    Row 0 is the constant term (frequency 0). For k in 1..(p-1)/2, row 2k-1 is cos(2*pi*k*a/p)
    and row 2k is sin(2*pi*k*a/p), each normalized to unit norm.
    """
    a = torch.arange(p, dtype=torch.float32)
    basis = torch.zeros(p, p)
    basis[0] = 1.0 / math.sqrt(p)
    for k in range(1, p // 2 + 1):
        cos = torch.cos(2 * math.pi * k * a / p)
        sin = torch.sin(2 * math.pi * k * a / p)
        basis[2 * k - 1] = cos / cos.norm()
        basis[2 * k] = sin / sin.norm()
    return basis


def freq_rows(k: int):
    """Basis row indices for frequency k."""
    return [0] if k == 0 else [2 * k - 1, 2 * k]


def filter_embedding(W_E: torch.Tensor, p: int, keep=None, remove=None) -> torch.Tensor:
    """Return a copy of W_E with the number-token rows reconstructed from a subset of frequencies.

    `remove`: zero the listed frequencies. `keep`: zero every frequency except the listed ones
    (the constant term is always kept). Exactly one of keep/remove should be set.
    """
    F = make_fourier_basis(p)
    num = W_E[:p]
    coeffs = F @ num  # (p, d_model): one coefficient vector per basis row
    mask = torch.ones(p, 1)
    if remove is not None:
        for k in remove:
            for r in freq_rows(k):
                mask[r] = 0.0
    elif keep is not None:
        mask = torch.zeros(p, 1)
        mask[0] = 1.0  # keep the constant
        for k in keep:
            for r in freq_rows(k):
                mask[r] = 1.0
    recon = F.t() @ (coeffs * mask)
    out = W_E.clone()
    out[:p] = recon
    return out


def phase_shift_embedding(W_E: torch.Tensor, p: int, freq: int, delta: float) -> torch.Tensor:
    """Rotate the frequency-`freq` component of every number token by phase `delta`.

    A phase delta = 2*pi*freq/p corresponds to shifting the input value by +1, which (for a key
    frequency) shifts the model's answer by +1. This is the directional test in the contract.
    """
    F = make_fourier_basis(p)
    num = W_E[:p]
    r_cos, r_sin = freq_rows(freq)
    rc, rs = F[r_cos], F[r_sin]
    c = rc @ num  # (d_model,)
    s = rs @ num  # (d_model,)
    shifted_cos = math.cos(delta) * rc - math.sin(delta) * rs
    shifted_sin = math.sin(delta) * rc + math.cos(delta) * rs
    delta_num = torch.outer(shifted_cos - rc, c) + torch.outer(shifted_sin - rs, s)
    out = W_E.clone()
    out[:p] = num + delta_num
    return out
