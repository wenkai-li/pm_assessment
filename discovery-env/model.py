"""A small 1-layer transformer for modular addition, in the style of the grokking literature.

No layer norm; learned positional embeddings; the forward pass can take either token ids or a
precomputed `input_embeds` tensor, which is what the intervention library uses to inject ablated
or phase-shifted embeddings.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 3
    seed: int = 0

    @property
    def d_vocab(self) -> int:
        return self.p + 1  # the '=' token is index p

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        s = cfg.d_model ** -0.5
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * s)
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * s)
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * s)
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("bnd,hde->bhne", x, self.W_Q)
        k = torch.einsum("bnd,hde->bhne", x, self.W_K)
        v = torch.einsum("bnd,hde->bhne", x, self.W_V)
        scores = torch.einsum("bhne,bhme->bhnm", q, k) / math.sqrt(self.cfg.d_head)
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhnm,bhme->bhne", attn, v)
        return torch.einsum("bhne,hed->bnd", z, self.W_O)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) * cfg.d_model ** -0.5)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) * cfg.d_mlp ** -0.5)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x @ self.W_in + self.b_in)
        return h @ self.W_out + self.b_out


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.W_E = nn.Parameter(torch.randn(cfg.d_vocab, cfg.d_model) * cfg.d_model ** -0.5)
        self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * cfg.d_model ** -0.5)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.d_vocab) * cfg.d_model ** -0.5)

    def forward(self, tokens: torch.Tensor = None, input_embeds: torch.Tensor = None) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.W_E[tokens]
        x = input_embeds + self.W_pos[: input_embeds.shape[1]]
        x = x + self.attn(x)
        x = x + self.mlp(x)
        logits = x @ self.W_U
        return logits[:, -1, :]  # logits at the '=' position
