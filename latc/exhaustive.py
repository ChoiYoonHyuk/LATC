from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from .affine import parity_table, validate_dimension, validate_noise

POOLS = {"short": (0, 1, 2, 3, 4, 8), "long": (0, 1, 2, 4, 8, 15)}


@lru_cache(maxsize=10)
def history_metadata(n: int):
    if not 1 <= n <= 12:
        raise ValueError("exhaustive solver supports 1..12 total eligible identities")
    powers = np.array([3**j for j in range(n)], dtype=np.int64)
    states = np.arange(3**n, dtype=np.int64)
    counts = np.zeros(len(states), dtype=np.int8)
    masks = np.zeros(len(states), dtype=np.int32)
    for j, p in enumerate(powers):
        observed = (states // p) % 3 != 0
        counts += observed
        masks += observed.astype(np.int32) * (1 << j)
    return powers, counts, masks


def regional_history_law(d: int, pool: tuple[int, ...], eta: float):

    validate_dimension(d)
    validate_noise(eta)
    m, k = 1 << d, len(pool)
    if not pool or len(set(pool)) != k or any(x < 0 or x >= m for x in pool):
        raise ValueError("pool must contain distinct valid coordinates")
    if k > 6:
        raise ValueError("at most six eligible identities per region")
    slopes = np.tile(np.arange(m), 2)
    beta = np.repeat(np.arange(2), m)
    clean = parity_table(d)[slopes[:, None] & np.arange(m)[None, :]] ^ beta[:, None]
    signs = 1.0 - 2.0 * clean
    powers, counts, masks = history_metadata(k)
    likelihoods = np.ones((3**k, 2*m), dtype=np.float64)
    for h in range(1, 3**k):
        for j, power in enumerate(powers):
            digit = (h // power) % 3
            if digit:
                prev = h - digit*power
                channel = np.where(clean[:, pool[j]] == digit-1, 1-eta, eta)
                likelihoods[h] = likelihoods[prev] * channel
                break
    mass = likelihoods.mean(axis=1)
    signed_mass = likelihoods @ signs / (2*m)
    for j, x in enumerate(pool):
        signed_mass[(masks & (1 << j)) != 0, x] = 0.0
    unnormalized_energy = np.einsum("ij,ij->i", signed_mass, signed_mass)
    reference = np.exp2(-counts.astype(np.float64))
    return mass, reference, unnormalized_energy


def history_costs(d: int, pool: tuple[int, ...], eta: float, prior: float = 0.5):
    if not 0 <= prior <= 1:
        raise ValueError("prior must lie in [0,1]")
    mass, ref, energy = regional_history_law(d, pool, eta)
    k, population = len(pool), 2*(1 << d)
    _, count_reg, _ = history_metadata(k)

    probability = prior*ref[:, None]*mass[None, :] + (1-prior)*mass[:, None]*ref[None, :]
    total_count = count_reg[:, None] + count_reg[None, :]
    numerator = (prior**2*ref[:, None]**2*energy[None, :] +
                 (1-prior)**2*energy[:, None]*ref[None, :]**2)
    gain = np.divide(numerator, probability, out=np.zeros_like(probability), where=probability > 0)
    cost = (probability*(population-total_count) - (1-2*eta)**2*gain)/(4*population)
    if cost.min() < -1e-13:
        raise ArithmeticError("negative history-weighted variance")
    return np.maximum(cost, 0.0).ravel(), probability.ravel()


@njit(cache=True)
def _fixed_continuations(cost, n, powers):
    states_count = len(cost)
    F = np.full((n+1, states_count), np.inf)

    choices = np.zeros((n+1, states_count), dtype=np.uint16)
    pair_visits = 0
    for final_mask in range(1 << n):
        positions = np.empty(n, dtype=np.int64)
        k = 0
        for j in range(n):
            if final_mask & (1 << j):
                positions[k] = j
                k += 1
        size = 3**k
        mapping = np.empty(size, dtype=np.int64)
        mapping[0] = 0
        length = 1
        for j in range(k):
            power = powers[positions[j]]
            for old in range(length):
                mapping[old+length] = mapping[old]+power
                mapping[old+2*length] = mapping[old]+2*power
            length *= 3
        marginal = np.empty(size, dtype=np.float64)
        for local_h in range(size-1, -1, -1):
            remainder = local_h
            missing_power = 0
            local_power = 1
            for j in range(k):
                if remainder % 3 == 0:
                    missing_power = local_power
                    break
                remainder //= 3
                local_power *= 3
            global_h = mapping[local_h]
            if missing_power == 0:
                value = cost[global_h]
            else:
                value = marginal[local_h+missing_power] + marginal[local_h+2*missing_power]
            marginal[local_h] = value
            if value < F[k, global_h]:
                F[k, global_h] = value
                choices[k, global_h] = final_mask
            pair_visits += 1
    return F, choices, pair_visits


@njit(cache=True)
def _adaptive_all(cost, n, powers, counts):
    roots = np.zeros(n+1)
    first_actions = np.full(n+1, -1, dtype=np.int16)
    for budget in range(n+1):
        value = np.empty(len(cost), dtype=np.float64)
        for h in range(len(cost)-1, -1, -1):
            count = counts[h]
            if count == budget:
                value[h] = cost[h]
            elif count < budget:
                best = np.inf
                best_j = -1
                for j in range(n):
                    power = powers[j]
                    if (h // power) % 3 == 0:
                        candidate = value[h+power] + value[h+2*power]
                        if candidate < best:
                            best, best_j = candidate, j
                value[h] = best
                if h == 0:
                    first_actions[budget] = best_j
        roots[budget] = value[0]
    return roots, first_actions


@dataclass
class ExhaustiveResult:
    d: int
    pool: tuple[int, ...]
    eta: float
    prior: float
    fixed: np.ndarray
    two_batch: np.ndarray
    adaptive: np.ndarray
    fixed_masks: np.ndarray
    first_batch_masks: np.ndarray
    first_adaptive_actions: np.ndarray
    max_normalization_error: float
    joint_states: int
    marginal_pairs: int

    def rows(self, pool_name: str = "custom"):
        population = 2*(1 << self.d)
        for b in range(len(self.fixed)):
            factor = population/(population-b) if b < population else math.nan
            yield {"pool": pool_name, "d": self.d, "eta": self.eta, "budget": b,
                   "fixed_full": float(self.fixed[b]), "two_batch_full": float(self.two_batch[b]),
                   "adaptive_full": float(self.adaptive[b]),
                   "fixed_unqueried": float(self.fixed[b]*factor),
                   "two_batch_unqueried": float(self.two_batch[b]*factor),
                   "adaptive_unqueried": float(self.adaptive[b]*factor),
                   "fixed_mask": int(self.fixed_masks[b]),
                   "first_batch_mask": int(self.first_batch_masks[b]),
                   "first_batch_size": int(self.first_batch_masks[b]).bit_count(),
                   "first_adaptive_action": int(self.first_adaptive_actions[b]),
                   "joint_states": self.joint_states, "marginal_pairs": self.marginal_pairs,
                   "normalization_error": self.max_normalization_error}


def solve_exhaustive(d: int = 4, pool: tuple[int, ...] = POOLS["short"],
                     eta: float = 0.05, prior: float = 0.5,
                     return_continuations: bool = False):

    pool = tuple(int(x) for x in pool)
    n = 2*len(pool)
    powers, counts, masks = history_metadata(n)
    cost, probability = history_costs(d, pool, eta, prior)
    sums = np.bincount(masks, weights=probability, minlength=1 << n)
    normalization_error = float(np.max(np.abs(sums-1)))
    if normalization_error > 1e-10:
        raise ArithmeticError("partial-history probabilities do not normalize")
    F, choices, pair_visits = _fixed_continuations(cost, n, powers)
    fixed = F[:, 0].copy()
    adaptive, first_actions = _adaptive_all(cost, n, powers, counts)
    two_batch = np.empty(n+1)
    first_masks = np.zeros(n+1, dtype=np.uint16)
    subset_counts = np.array([mask.bit_count() for mask in range(1 << n)])
    for b in range(n+1):
        valid = counts <= b
        by_first_set = np.bincount(masks[valid], weights=F[b, valid], minlength=1 << n)
        by_first_set[subset_counts > b] = np.inf
        first_masks[b] = np.argmin(by_first_set)
        two_batch[b] = by_first_set[first_masks[b]]
    tolerance = 2e-11
    if np.any(adaptive > two_batch+tolerance) or np.any(two_batch > fixed+tolerance):
        raise ArithmeticError("R_ad <= R_2 <= R_fixed violated")
    for values in (fixed, two_batch, adaptive):
        if np.any(np.diff(values) > tolerance):
            raise ArithmeticError("expected FULL-population Bayes risk increased")
    result = ExhaustiveResult(d, pool, eta, prior, fixed, two_batch, adaptive,
                              choices[:, 0].copy(), first_masks, first_actions,
                              normalization_error, len(cost), pair_visits)
    return (result, F, choices, cost) if return_continuations else result
