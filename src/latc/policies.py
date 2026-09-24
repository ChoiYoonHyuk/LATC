"""Algorithm 1, fixed controls, and persistent synthetic populations."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .algebra import affine_values, likelihood_table, outcome_code
from .posterior import Prediction, predict_two_regions
from .pools import PoolDesign


@dataclass(frozen=True)
class Acquisition:
    queries: tuple[np.ndarray, np.ndarray]
    labels: tuple[np.ndarray, np.ndarray]
    selected_region: int | None  # zero-based in files and Python
    first_likelihood: float | None
    prediction: Prediction


def generate_population(d: int, eta: float, rng: np.random.Generator,
                        prior: float = 0.5) -> tuple[np.ndarray, int, int]:
    if not 0 <= eta < 0.5 or not 0 <= prior <= 1:
        raise ValueError("Invalid noise or regional prior")
    m = 1 << d
    active = 0 if rng.random() < prior else 1
    theta = int(rng.integers(2 * m))
    labels = rng.integers(0, 2, size=(2, m), dtype=np.uint8)
    labels[active] = affine_values(d, theta) ^ (rng.random(m) < eta).astype(np.uint8)
    return labels, active, theta


def prescribed_two_batch(d: int, eta: float, prior: float, design: PoolDesign,
                         labels: np.ndarray, budget: int, tie_tolerance: float = 1e-12) -> Acquisition:
    """Threshold ONE, not prior odds; both blocks are excluded from continuation.

    Region two's designated block is never acquired in the first batch. All
    verification labels count toward the same total budget. Terminal inference
    nevertheless uses the supplied public prior and every acquired observation.
    """
    design.validate()
    t = len(design.blocks[0])
    if budget < t or any(budget - t > len(c) for c in design.remainders):
        raise ValueError("The exact two-batch budget is infeasible")
    if labels.shape != (2, 1 << d):
        raise ValueError("Wrong persistent population shape")
    first = design.blocks[0]
    likelihood = float(likelihood_table(first, eta)[outcome_code(labels[0, first])])
    selected = 0 if likelihood >= 1 - tie_tolerance else 1
    queries = [first.copy(), np.array([], dtype=np.int64)]
    queries[selected] = np.concatenate((queries[selected], design.remainders[selected][:budget - t]))
    return _finish(d, eta, prior, design, labels, queries, budget, selected, likelihood)


def fixed_control(d: int, eta: float, prior: float, design: PoolDesign,
                  labels: np.ndarray, budget: int, balanced: bool = False) -> Acquisition:
    """L.2 controls acquire from C_j, NOT from the designated blocks."""
    if balanced:
        counts = (budget // 2, budget - budget // 2)
    else:
        counts = (budget, 0) if prior >= 0.5 else (0, budget)
    if any(n < 0 or n > len(c) for n, c in zip(counts, design.remainders)):
        raise ValueError("Fixed-control budget exceeds a public remainder")
    queries = [c[:n].copy() for n, c in zip(counts, design.remainders)]
    return _finish(d, eta, prior, design, labels, queries, budget, None, None)


def _finish(d, eta, prior, design, labels, queries, budget, selected, likelihood) -> Acquisition:
    if sum(map(len, queries)) != budget:
        raise ArithmeticError("Incorrect acquired-label count")
    for q, pool in zip(queries, design.pools):
        if len(np.unique(q)) != len(q) or not np.isin(q, pool).all():
            raise ArithmeticError("Acquisition violated distinctness or pool membership")
    observed = tuple(labels[j, q].copy() for j, q in enumerate(queries))
    prediction = predict_two_regions(d, queries, observed, eta, prior)
    return Acquisition(tuple(queries), observed, selected, likelihood, prediction)


def indexed_rng(seed: int, study: int, d: int, eta: float, replicate: int, component: int
                ) -> np.random.Generator:
    """Explicit, schedule-independent PCG64 stream map, documented in REPRODUCIBILITY.

    Noise is represented by a micro-probability integer; reject values not exactly
    represented at that resolution instead of introducing stream-key collisions.
    """
    code = int(round(eta * 1_000_000))
    if not math.isclose(eta, code / 1_000_000, abs_tol=1e-14, rel_tol=0):
        raise ValueError("Stream map supports noise rates with at most six decimal places")
    key = [seed, study, d, code, replicate, component]
    if min(key) < 0:
        raise ValueError("Stream indices must be nonnegative")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
