from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def validate_dimension(d: int) -> None:
    if not isinstance(d, (int, np.integer)) or d < 1 or d > 22:
        raise ValueError("d must be an integer in [1, 22]; exact arrays scale as 2**d")


def validate_noise(eta: float) -> None:
    if not np.isfinite(eta) or not 0 <= eta < 0.5:
        raise ValueError("eta must be finite and in [0, 0.5)")


def gf2_rank(rows) -> int:

    pivots: dict[int, int] = {}
    for item in rows:
        value = int(item)
        if value < 0:
            raise ValueError("GF(2) rows must be nonnegative")
        while value:
            lead = value.bit_length() - 1
            if lead in pivots:
                value ^= pivots[lead]
            else:
                pivots[lead] = value
                break
    return len(pivots)


def augmented_rank(coordinates) -> int:
    return gf2_rank((int(x) << 1) | 1 for x in coordinates)


@lru_cache(maxsize=16)
def parity_table(d: int) -> NDArray[np.int8]:
    validate_dimension(d)
    result = np.fromiter((i.bit_count() & 1 for i in range(1 << d)),
                         dtype=np.int8, count=1 << d)
    result.flags.writeable = False
    return result


def fwht(values: NDArray) -> FloatArray:

    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim == 0:
        raise ValueError("fwht expects at least one array axis")
    n = result.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("last-axis length must be a positive power of two")
    step = 1
    while step < n:
        blocks = result.reshape(*result.shape[:-1], -1, 2 * step)
        left = blocks[..., :step].copy()
        right = blocks[..., step:].copy()
        blocks[..., :step] = left + right
        blocks[..., step:] = left - right
        step *= 2
    return result


def clean_labels(d: int, beta: int, slope: int, coordinates=None) -> NDArray[np.int8]:
    validate_dimension(d)
    if beta not in (0, 1) or not 0 <= slope < (1 << d):
        raise ValueError("invalid affine task")
    xs = np.arange(1 << d, dtype=np.int64) if coordinates is None else np.asarray(coordinates, dtype=np.int64)
    if np.any(xs < 0) or np.any(xs >= (1 << d)):
        raise ValueError("coordinate is outside the binary cube")
    return parity_table(d)[np.bitwise_and(xs, slope)] ^ beta


def validate_observations(d: int, coordinates, labels) -> tuple[IntArray, NDArray[np.int8]]:
    validate_dimension(d)
    xs = np.asarray(coordinates, dtype=np.int64)
    raw = np.asarray(labels)
    if xs.ndim != 1 or raw.ndim != 1 or len(xs) != len(raw):
        raise ValueError("coordinates and labels must be equally sized vectors")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("targets must be binary")
    if np.any(xs < 0) or np.any(xs >= 1 << d):
        raise ValueError("queried coordinate outside population")
    if len(np.unique(xs)) != len(xs):
        raise ValueError("a fixed identity may be queried only once")
    return xs, raw.astype(np.int8)


@dataclass(frozen=True)
class RegionalPosterior:
    log_lr: float
    clean_sign: FloatArray
    query_count: int
    rank: int


def regional_posterior(d: int, eta: float, coordinates, labels) -> RegionalPosterior:

    validate_noise(eta)
    xs, ys = validate_observations(d, coordinates, labels)
    n, m = len(xs), 1 << d
    rank = augmented_rank(xs)
    signed = np.zeros(m, dtype=np.float64)
    signed[xs] = 1 - 2 * ys
    correlations = fwht(signed)
    errors = np.vstack(((n - correlations) / 2, (n + correlations) / 2))
    if eta == 0:
        log_weights = np.where(errors == 0, 0.0, -np.inf)
    else:
        log_weights = (n - errors) * np.log1p(-eta) + errors * math.log(eta)
    normalizer = float(logsumexp(log_weights))
    if not np.isfinite(normalizer):
        return RegionalPosterior(-math.inf, np.zeros(m), n, rank)
    weights = np.exp(log_weights - normalizer)
    means = fwht(weights[0] - weights[1])
    means = np.clip(means, -1.0, 1.0)

    log_lr = 0.0 if rank == n else normalizer - math.log(2 * m) + n * math.log(2)
    return RegionalPosterior(float(log_lr), means, n, rank)


def dense_regional_posterior(d: int, eta: float, coordinates, labels,
                             evaluation_coordinates=None) -> RegionalPosterior:

    validate_noise(eta)
    xs, ys = validate_observations(d, coordinates, labels)
    n, m = len(xs), 1 << d
    evaluation = np.arange(m) if evaluation_coordinates is None else np.asarray(evaluation_coordinates, dtype=np.int64)
    slopes = np.tile(np.arange(m), 2)
    offsets = np.repeat(np.arange(2), m)
    query_labels = parity_table(d)[slopes[:, None] & xs[None, :]] ^ offsets[:, None]
    errors = np.sum(query_labels != ys[None, :], axis=1)
    if eta == 0:
        logw = np.where(errors == 0, 0.0, -np.inf)
    else:
        logw = (n - errors) * np.log1p(-eta) + errors * math.log(eta)
    normalizer = float(logsumexp(logw))
    rank = augmented_rank(xs)
    if not np.isfinite(normalizer):
        return RegionalPosterior(-math.inf, np.zeros(len(evaluation)), n, rank)
    signs = 1 - 2 * (parity_table(d)[slopes[:, None] & evaluation[None, :]] ^ offsets[:, None])
    means = np.exp(logw - normalizer) @ signs
    return RegionalPosterior(normalizer - math.log(2 * m) + n * math.log(2), means, n, rank)


def reliability_probabilities(posteriors: tuple[RegionalPosterior, RegionalPosterior],
                              prior: float = 0.5) -> FloatArray:
    if not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("prior must be in [0, 1]")
    lp = np.array([math.log(prior) if prior else -math.inf,
                   math.log1p(-prior) if prior < 1 else -math.inf])
    log_weights = lp + np.array([p.log_lr for p in posteriors])
    norm = float(logsumexp(log_weights))
    if not np.isfinite(norm):
        raise ValueError("zero-probability transcript under the exclusive reliability law")
    return np.exp(log_weights - norm)


def posterior_prediction(d: int, eta: float, queries, observed_labels,
                         prior: float = 0.5):

    if len(queries) != 2 or len(observed_labels) != 2:
        raise ValueError("the main experiments have exactly two regions")
    posts = tuple(regional_posterior(d, eta, queries[j], observed_labels[j]) for j in range(2))
    gamma = reliability_probabilities(posts, prior)
    predictions = np.vstack([0.5 - 0.5 * gamma[j] * (1 - 2 * eta) * posts[j].clean_sign
                              for j in range(2)])
    for j in range(2):
        predictions[j, np.asarray(queries[j], dtype=np.int64)] = observed_labels[j]
    return predictions, posts, gamma


@dataclass(frozen=True)
class TargetArray:
    d: int
    eta: float
    seed: int
    reliable_region: int
    beta: int
    slope: int
    labels: NDArray[np.int8]


def sample_target_array(d: int, eta: float, seed: int) -> TargetArray:

    validate_dimension(d)
    validate_noise(eta)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    j = int(rng.integers(0, 2))
    beta = int(rng.integers(0, 2))
    slope = int(rng.integers(0, 1 << d))
    flips = (rng.random(1 << d) < eta).astype(np.int8)
    labels = np.empty((2, 1 << d), dtype=np.int8)
    labels[j] = clean_labels(d, beta, slope) ^ flips
    labels[1 - j] = rng.integers(0, 2, size=1 << d, dtype=np.int8)
    labels.flags.writeable = False
    return TargetArray(d, eta, int(seed), j, beta, slope, labels)
