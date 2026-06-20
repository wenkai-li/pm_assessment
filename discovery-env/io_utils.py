"""Save / load helpers for trained checkpoints."""
from dataclasses import asdict

import torch

from model import Config, Transformer


def save_model(model: Transformer, path: str) -> None:
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict()}, path)


def load_model(path: str) -> Transformer:
    # weights_only=True avoids unpickling arbitrary objects; cfg is a plain dict of primitives.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(Config(**blob["cfg"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model
