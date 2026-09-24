"""Theorem-scale subcube-only testing simulation, Appendix Q.

No ambient cube is materialized. Adding an affine restriction leaves the
likelihood invariant, so Bernoulli errors around the zero restriction exactly
integrate the uniform affine parameter.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from .algebra import fwht
from .bounds import subcube_size, entropy, _integer_budget
from .intervals import IntervalArithmetic
from .policies import indexed_rng
from .statistics import equal_prior_error_interval


def block_log_likelihood(labels: np.ndarray, eta: float) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 2 or not np.isin(y, [0, 1]).all() or not 0 < eta < .5:
        raise ValueError("Require binary outcome rows and fixed positive noise below 1/2")
    t = y.shape[-1]
    if t == 0 or t & (t-1):
        raise ValueError("A full subcube has a power-of-two number of labels")
    correlations = fwht(1 - 2*y.astype(np.int64))
    e = np.stack(((t-correlations)//2, (t+correlations)//2), axis=1)
    log_weights = (t-e)*math.log1p(-eta) + e*math.log(eta)
    return t*math.log(2) - math.log(2*t) + logsumexp(log_weights, axis=(1,2))


def simulate_verification(r: int, eta: float, samples: int = 131072,
                          seed: int = 2026, chunk_size: int = 512, study_id: int = 4096) -> dict:
    if samples <= 0 or chunk_size <= 0 or not 0 < eta < .5:
        raise ValueError("Invalid verification experiment")
    t = subcube_size(r)
    streams = [indexed_rng(seed, study_id, r-1, eta, 0, c) for c in (1,2)]
    k_affine = k_independent = 0
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples-start)
        affine = (streams[0].random((count,t)) < eta).astype(np.uint8)
        independent = streams[1].integers(0,2,size=(count,t),dtype=np.uint8)
        # Equivalent to L >= 1 - 1e-12; the prescribed tie selects region one.
        threshold = math.log1p(-1e-12)
        k_affine += int(np.count_nonzero(block_log_likelihood(affine,eta) < threshold))
        k_independent += int(np.count_nonzero(block_log_likelihood(independent,eta) >= threshold))
    arithmetic = IntervalArithmetic(80)
    b = _integer_budget(r, arithmetic.number(str(eta)))
    return dict(r=r, q=r*r, eta=eta, t=t, b=b, continuation=b-t,
                capacity_margin=(b-t)*(1-entropy(eta))/r-1,
                seed=seed, study_id=study_id, chunk_size=chunk_size,
                **equal_prior_error_interval(k_affine,k_independent,samples))
