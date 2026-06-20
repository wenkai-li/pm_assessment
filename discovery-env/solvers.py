"""Scripted solver policies with KNOWN behavior, used to validate that the diagnostic protocol
separates premature commitment from a capability ceiling. A real LLM solver plugs in behind the same
interface (reset / act / observe / observe_rejected).

Policy interface:
  reset(inst, hint, attempt_seed)   start an episode; `hint` correct leading keys are given for free.
  act(verified_depth) -> ("probe", prefix) | ("submit", chain)
  observe(prefix, ok)               feedback from a probe.
  observe_rejected()                called when a submit is refused under forced continuation.
"""
import random


class Thorough:
    """Descends level by level to the required depth, then commits. Solves single-shot."""
    def reset(self, inst, hint, attempt_seed):
        self.D, self.B = inst["depth"], inst["branching"]
        self.known = list(inst["hidden"][:hint])
        self.cand = 0

    def act(self, verified):
        if len(self.known) >= self.D:
            return ("submit", self.known)
        return ("probe", self.known + [self.cand])

    def observe(self, prefix, ok):
        if ok and len(prefix) == len(self.known) + 1:
            self.known.append(prefix[-1])
            self.cand = 0
        else:
            self.cand += 1

    def observe_rejected(self):
        pass


class Premature:
    """Can do each step, but commits at a random depth and guesses the rest. Under forced
    continuation it keeps going and solves. This is the premature-commitment signature: low pass@1,
    high pass@k, solves when not allowed to commit early."""
    def reset(self, inst, hint, attempt_seed):
        self.D, self.B = inst["depth"], inst["branching"]
        self.known = list(inst["hidden"][:hint])
        self.cand = 0
        self.forced = False
        self._rng = random.Random(1000 + attempt_seed * 7919 + inst["seed"])
        self.stop_depth = self._rng.randint(max(hint, 1), self.D)

    def act(self, verified):
        if len(self.known) >= self.D:
            return ("submit", self.known)
        if not self.forced and len(self.known) >= self.stop_depth:
            guess = list(self.known) + [self._rng.randrange(self.B)
                                        for _ in range(self.D - len(self.known))]
            return ("submit", guess)
        return ("probe", self.known + [self.cand])

    def observe(self, prefix, ok):
        if ok and len(prefix) == len(self.known) + 1:
            self.known.append(prefix[-1])
            self.cand = 0
        else:
            self.cand += 1

    def observe_rejected(self):
        self.forced = True  # keep exploring when early commit is refused


class CapabilityLimited:
    """Cannot use the depth-gated feedback, so it never advances. Fails even with forced
    continuation and even when given all but the last key. This is the capability-ceiling signature."""
    def reset(self, inst, hint, attempt_seed):
        self.D, self.B = inst["depth"], inst["branching"]
        self.known = list(inst["hidden"][:hint])
        self.cand = 0

    def act(self, verified):
        if len(self.known) >= self.D:
            return ("submit", self.known)
        return ("probe", self.known + [self.cand % self.B])

    def observe(self, prefix, ok):
        self.cand += 1  # ignores `ok`: never advances, cannot do the reasoning step

    def observe_rejected(self):
        pass
