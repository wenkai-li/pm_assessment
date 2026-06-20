"""A depth-controlled, depth-gated task used to validate the exploration diagnostic.

The solution is a hidden chain of d* keys. Information is gated by depth: you can only verify key j
once keys 1..j-1 are correct, so the answer genuinely sits at depth d* in a branching search tree of
width B. You cannot guess your way down; you have to descend. This makes premature commitment
verifiable (committing before descending to d* means committing keys you could not have known) and
the required depth d* exactly known.

This is a synthetic instrument, like a controlled testbed for the diagnostic protocol. The same
protocol wraps the real ML-research domains, where depth is a circuit's compositional depth, a
multi-hop dependency, or a chain of dependent experiments.
"""
import random


class LatentChainTask:
    def __init__(self, depth=8, branching=4):
        self.depth = depth
        self.branching = branching

    def build(self, seed):
        rng = random.Random(seed)
        hidden = [rng.randrange(self.branching) for _ in range(self.depth)]
        return {"hidden": hidden, "depth": self.depth, "branching": self.branching, "seed": seed}

    def required_depth(self, inst):
        return inst["depth"]

    def on_path(self, inst, prefix):
        """The depth-gated experiment: is `prefix` a correct prefix of the hidden chain?"""
        h = inst["hidden"]
        return len(prefix) <= len(h) and list(prefix) == h[: len(prefix)]

    def verify(self, inst, answer):
        """The terminal verifier (no language-model judge): exact match to the hidden chain."""
        return list(answer) == inst["hidden"]

    def correct_depth(self, inst, answer):
        """Length of the longest correct leading prefix of `answer` (the depth to which the
        committed answer is actually right). Used by the depth-calibrated judge."""
        h = inst["hidden"]
        d = 0
        for i, key in enumerate(answer):
            if i < len(h) and key == h[i]:
                d += 1
            else:
                break
        return d
