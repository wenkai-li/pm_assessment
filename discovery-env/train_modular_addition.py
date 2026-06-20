"""Train the modular-addition transformer to grokking for one or more seeds.

Full-batch training with weight decay is the standard recipe that induces the Fourier algorithm.
Each seed lands on a different set of key frequencies, which is the per-instance randomization the
environment relies on.
"""
import argparse
import os

import torch

from interventions import accuracy, all_pairs, make_batch
from io_utils import save_model
from model import Config, Transformer


def train_one(seed: int, p: int = 113, steps: int = 20000, train_frac: float = 0.3, lr: float = 1e-3,
              wd: float = 1.0, device: str = "cpu") -> Transformer:
    cfg = Config(p=p, seed=seed)
    model = Transformer(cfg).to(device)

    pairs = all_pairs(p)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    n_train = int(train_frac * len(pairs))
    train_pairs = [pairs[i] for i in perm[:n_train]]
    test_pairs = [pairs[i] for i in perm[n_train:]]
    tr_tok, tr_tgt = make_batch(train_pairs, p)
    te_tok, te_tgt = make_batch(test_pairs, p)
    tr_tok, tr_tgt = tr_tok.to(device), tr_tgt.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    for step in range(steps):
        opt.zero_grad()
        logits = model(tr_tok)
        loss = torch.nn.functional.cross_entropy(logits, tr_tgt)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                te_acc = accuracy(model(te_tok.to(device)), te_tgt.to(device))
            print(f"seed {seed} step {step:6d} loss {loss.item():.4f} test_acc {te_acc:.3f}", flush=True)
            if te_acc > 0.995:
                print(f"seed {seed} grokked at step {step}", flush=True)
                break
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for s in args.seeds:
        model = train_one(s, steps=args.steps)
        path = os.path.join(args.out, f"seed_{s}.pt")
        save_model(model, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
