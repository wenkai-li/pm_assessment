"""Domain 3 — Data forensics and attribution, implemented as a runnable second worked instance.

It shows that the depth ladder and the depth-calibrated judge are not specific to interpretability:
the same machinery grades a different research task. Everything here is pure numpy, so it runs on CPU
with no training infrastructure.

System S: a linear classifier trained on a dataset that carries a hidden spurious correlation.
Hidden H: one spurious feature index s*, randomized per instance, that predicts the label in the
  training distribution but is independent of it in the counterfactual (decorrelated) distribution.
  The label is driven by a noisy causal feature, so the model is tempted to lean on the cleaner
  spurious feature, and its accuracy drops on the counterfactual distribution.
The agent must discover the spurious reliance and prove understanding by predicting the effect of
counterfactual interventions on data it has not seen.

Derivation ladder (shallow to deep), each rung checked on fresh held-out samples:
  detect       there is a gap: predict the counterfactual accuracy, which sits well below train.
  localize     which feature: permuting the claimed spurious feature drops training accuracy as
               predicted (necessity of the named feature).
  specificity  only that feature: permuting other features barely moves accuracy.
  quantify     how much it relies: predict the counterfactual accuracy to a tight tolerance.
  extrapolate  predict accuracy at an unseen correlation strength rho (the directional curve), which
               only a correct account of the reliance gets right.
A deep rung cannot hold without the shallow ones: the counterfactual accuracy and the rho curve cannot
be predicted without correctly localizing the spurious feature.
"""
import numpy as np

from contract import grade_ladder
from domains.base import Domain

D_FEATURES = 8
N_TRAIN = 3000
N_EVAL = 1500
CAUSAL_NOISE = 1.1     # label noise on the causal feature (makes the spurious feature tempting)
SPURIOUS_MARGIN = 2.2  # how strongly the spurious feature separates the classes when rho = 1


def _sample(rng, n, causal, spurious, rho):
    """Sample (X, y). The causal feature drives a noisy label; the spurious feature separates the
    classes with strength rho (rho=1 train-like, rho=0 counterfactual)."""
    X = rng.standard_normal((n, D_FEATURES))
    y = (X[:, causal] + CAUSAL_NOISE * rng.standard_normal(n) > 0).astype(np.float64)
    X[:, spurious] = rho * (2 * y - 1) * SPURIOUS_MARGIN + rng.standard_normal(n)
    return X, y


def _fit_logreg(X, y, steps=300, lr=0.2):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y))
        b -= lr * g.mean()
    return w, b


def _accuracy(model, X, y):
    w, b = model
    return float((((X @ w + b) > 0).astype(np.float64) == y).mean())


class DataForensicsDomain(Domain):
    name = "data_forensics"
    DERIVATION_LADDER = ("detect", "localize", "specificity", "quantify", "extrapolate")

    def build_instance(self, seed):
        rng = np.random.default_rng(seed)
        causal = int(rng.integers(0, D_FEATURES))
        spurious = int(rng.integers(0, D_FEATURES))
        while spurious == causal:
            spurious = int(rng.integers(0, D_FEATURES))
        Xtr, ytr = _sample(rng, N_TRAIN, causal, spurious, rho=1.0)
        model = _fit_logreg(Xtr, ytr)
        return {"model": model, "causal": causal, "spurious": spurious, "seed": seed,
                "symptom": "A classifier with high training accuracy whose accuracy drops on a "
                           "decorrelated test distribution. Find what it relies on."}

    def tools(self):
        return {"accuracy_at_rho", "permute_feature", "keep_only_features", "sample_batch",
                "train_surrogate", "checkpoint", "restore"}

    def required_depth(self, instance):
        return len(self.DERIVATION_LADDER)

    def heldout(self, instance, seed=987654):
        return {"rng_seed": seed}

    def _eval(self, instance, rho, seed, permute=None, keep_only=None):
        rng = np.random.default_rng(seed)
        X, y = _sample(rng, N_EVAL, instance["causal"], instance["spurious"], rho)
        if permute is not None:
            for j in permute:
                X[:, j] = rng.permutation(X[:, j])
        if keep_only is not None:
            mask = np.zeros(D_FEATURES, dtype=bool)
            mask[list(keep_only)] = True
            X[:, ~mask] = 0.0
        return _accuracy(instance["model"], X, y)

    def execute(self, pred, instance, structure, heldout):
        seed = heldout["rng_seed"]
        claimed = list(structure.get("spurious", structure.get("elements", [])))
        train_acc = self._eval(instance, rho=1.0, seed=seed + 1)

        counterfactual = self._eval(instance, rho=0.0, seed=seed)

        if pred["id"] == "detect":
            actual = counterfactual
            passed = (abs(actual - pred["predicted"]) <= pred["tol"]) and (train_acc - actual >= 0.08)
        elif pred["id"] == "localize":
            # A spurious feature has a clear signature: permuting it on the training distribution drops
            # accuracy a lot, but permuting it on the already-decorrelated counterfactual distribution
            # barely moves accuracy. The causal feature drops both; a noise feature drops neither. So
            # this isolates the spurious feature and rejects over-claiming (any causal or noise feature
            # in the claimed set breaks the signature).
            actual = self._eval(instance, rho=1.0, seed=seed, permute=claimed)
            perm_cf = self._eval(instance, rho=0.0, seed=seed + 3, permute=claimed)
            passed = ((abs(actual - pred["predicted"]) <= pred["tol"])
                      and (train_acc - actual >= 0.08)
                      and (counterfactual - perm_cf <= 0.04))
        elif pred["id"] == "specificity":
            # Every claimed feature must individually carry the spurious signature, and probed features
            # outside the claim must not, which enforces a complete and minimal claim.
            probe = pred["probe"]
            actual = [self._eval(instance, rho=1.0, seed=seed, permute=[j]) for j in probe]
            err = sum(abs(a - p) for a, p in zip(actual, pred["predicted"])) / len(actual)
            spurious_ok = True
            for j in probe:
                drop_tr = train_acc - self._eval(instance, rho=1.0, seed=seed, permute=[j])
                drop_cf = counterfactual - self._eval(instance, rho=0.0, seed=seed + 3, permute=[j])
                is_spurious = (drop_tr >= 0.08) and (drop_cf <= 0.04)
                if (j in claimed) != is_spurious:
                    spurious_ok = False
                    break
            passed = (err <= pred["tol"]) and spurious_ok
        elif pred["id"] == "quantify":
            actual = self._eval(instance, rho=0.0, seed=seed + 7)
            passed = abs(actual - pred["predicted"]) <= pred["tol"]
        elif pred["id"] == "extrapolate":
            actual = self._eval(instance, rho=pred["rho"], seed=seed + 13)
            passed = abs(actual - pred["predicted"]) <= pred["tol"]
        else:
            actual, passed = None, False
        return actual, passed

    def structure_size(self, structure):
        return len(structure.get("spurious", structure.get("elements", [])))

    def max_structure(self, instance):
        return D_FEATURES

    def grade_episode(self, instance, contract, heldout, probes=0):
        return grade_ladder(self, instance, contract, heldout, probes)

    def reference_contract(self, instance):
        """The env knows the spurious feature, so it can compute the true rung values."""
        s = instance["spurious"]
        h = self.heldout(instance)
        seed = h["rng_seed"]
        detect = self._eval(instance, rho=0.0, seed=seed)
        localize = self._eval(instance, rho=1.0, seed=seed, permute=[s])
        probe = sorted({s, instance["causal"], (s + 1) % D_FEATURES, (s + 2) % D_FEATURES})
        spec = [self._eval(instance, rho=1.0, seed=seed, permute=[j]) for j in probe]
        quantify = self._eval(instance, rho=0.0, seed=seed + 7)
        rho_mid = 0.5
        extrap = self._eval(instance, rho=rho_mid, seed=seed + 13)
        return {"components": {"spurious": [s]},
                "predictions": [
                    {"id": "detect", "predicted": round(detect, 3), "tol": 0.05},
                    {"id": "localize", "predicted": round(localize, 3), "tol": 0.05},
                    {"id": "specificity", "probe": probe,
                     "predicted": [round(x, 3) for x in spec], "tol": 0.06},
                    {"id": "quantify", "predicted": round(quantify, 3), "tol": 0.04},
                    {"id": "extrapolate", "rho": rho_mid, "predicted": round(extrap, 3), "tol": 0.06},
                ]}

    def calibration_contracts(self, instance):
        import copy
        ref = self.reference_contract(instance)
        empty = copy.deepcopy(ref)
        empty["components"]["spurious"] = []
        whole = copy.deepcopy(ref)
        whole["components"]["spurious"] = list(range(D_FEATURES))
        wrong = copy.deepcopy(ref)
        wrong["components"]["spurious"] = [(instance["spurious"] + 3) % D_FEATURES]
        return {"reference": ref, "empty": empty, "whole-feature-set": whole, "wrong-feature": wrong}
