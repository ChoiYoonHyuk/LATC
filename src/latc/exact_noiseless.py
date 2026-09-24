"""Full-access noiseless acquisition, Appendix H, Eqs. (147)--(157).

The algorithms integrate every reduced-state outcome; they do not use Monte
Carlo, greedy substitutes, or an affine-rank approximation to noisy inference.
Arithmetic is float64 except for exact integer subset counts in Uniform.
"""
from __future__ import annotations
from functools import lru_cache
from math import comb
from typing import NamedTuple
import numpy as np

Region = tuple[int, int, int]  # count, augmented rank, consistency
State = tuple[Region, Region]
EMPTY: State = ((0, 0, 1), (0, 0, 1))


def span_size(rank: int) -> int:
    return 0 if rank == 0 else 1 << (rank - 1)


class DesignValue(NamedTuple):
    population_risk: float
    plan: tuple[int, int, int, int]  # q1, delta_k1, q2, delta_k2


class SubcubeValue(NamedTuple):
    risk: float
    first_plan: tuple[int, int, int, int]


class NoiselessOptimizer:
    def __init__(self, d: int, prior: float = 0.5):
        if not isinstance(d, int) or not 0 <= d <= 60 or not 0 <= prior <= 1:
            raise ValueError("Require integer 0 <= d <= 60 and prior in [0,1]")
        self.d, self.r, self.m, self.prior = d, d + 1, 1 << d, float(prior)
        # Per-instance caches avoid retaining completed optimizers globally.
        self.fixed_continuation = lru_cache(maxsize=None)(self._fixed_continuation)
        self.posterior = lru_cache(maxsize=None)(self._posterior)
        self._options = lru_cache(maxsize=None)(self._options_uncached)

    def _posterior(self, state: State) -> float:
        likelihoods = [2.0 ** (n - k) if o else 0.0 for n, k, o in state]
        a, b = self.prior * likelihoods[0], (1 - self.prior) * likelihoods[1]
        if a + b == 0:
            raise ValueError("Impossible noiseless history")
        return a / (a + b)

    def terminal_loss(self, state: State) -> float:
        p = self.posterior(state)
        gain = sum(g * g * (span_size(k) - n) if o else 0.0
                   for (n, k, o), g in zip(state, (p, 1 - p)))
        return (2 * self.m - sum(s[0] for s in state) - gain) / (8 * self.m)

    def _validate_budget(self, budget: int) -> None:
        if not isinstance(budget, int) or not 0 <= budget <= 2 * self.m:
            raise ValueError("Budget must be an integer in [0,2m]")
        if budget > 128:
            raise ValueError("Reduced-state enumeration guard: budget <=128")

    def normalize(self, risk: float, budget: int, unqueried: bool = True) -> float:
        if not unqueried:
            return risk
        if budget == 2 * self.m:
            raise ValueError("Unqueried normalization has no remaining targets")
        return risk * (2 * self.m) / (2 * self.m - budget)

    def feasible_ranks(self, count: int) -> tuple[int, ...]:
        if count == 0:
            return (0,)
        if not 0 <= count <= self.m:
            return ()
        return tuple(k for k in range(1, min(count, self.r) + 1) if count <= span_size(k))

    def _options_uncached(self, region: Region, added: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k, o = region
        if added < 0 or n + added > self.m:
            return np.array([], dtype=int), np.array([]), np.array([])
        if not o:
            return np.array([0]), np.array([0.0]), np.array([0.0])
        increments = np.array([dk for dk in range(min(added, self.r - k) + 1)
                               if n + added <= span_size(k + dk)], dtype=int)
        consistency = np.exp2(increments.astype(float) - added)
        closure = np.array([span_size(k + int(dk)) - n - added for dk in increments], dtype=float)
        return increments, consistency, closure

    @staticmethod
    def _expected_gains(p: float, a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Eq. (153), conditional on the current posterior. a_j is the probability
        # that the fixed continuation remains consistent when region j is inactive.
        x, y = p * a2[None, :], (1 - p) * a1[:, None]
        den = x + y
        shape = np.broadcast_shapes(x.shape, y.shape)
        g1 = p * (1 - a2[None, :]) + np.divide(x * x, den, out=np.zeros(shape), where=den > 0)
        g2 = (1 - p) * (1 - a1[:, None]) + np.divide(y * y, den, out=np.zeros(shape), where=den > 0)
        return g1, g2

    def _fixed_continuation(self, state: State, budget: int) -> DesignValue:
        remaining = budget - sum(s[0] for s in state)
        if remaining < 0:
            raise ValueError("State is already beyond the requested total budget")
        if remaining == 0:
            return DesignValue(self.terminal_loss(state), (0, 0, 0, 0))
        p = self.posterior(state)
        best_gain, best_plan = -np.inf, None
        lo = max(0, remaining - (self.m - state[1][0]))
        hi = min(remaining, self.m - state[0][0])
        for q1 in range(lo, hi + 1):
            q2 = remaining - q1
            dk1, a1, h1 = self._options(state[0], q1)
            dk2, a2, h2 = self._options(state[1], q2)
            if not dk1.size or not dk2.size:
                continue
            g1, g2 = self._expected_gains(p, a1, a2)
            gain = h1[:, None] * g1 + h2[None, :] * g2
            index = int(np.argmax(gain))
            value = float(gain.flat[index])
            if value > best_gain:
                i, j = np.unravel_index(index, gain.shape)
                best_gain = value
                best_plan = (q1, int(dk1[i]), q2, int(dk2[j]))
        if best_plan is None:
            raise ValueError("No feasible fixed continuation")
        return DesignValue((2 * self.m - budget - best_gain) / (8 * self.m), best_plan)

    def fixed(self, budget: int, unqueried: bool = True) -> float:
        self._validate_budget(budget)
        return self.normalize(self.fixed_continuation(EMPTY, budget).population_risk, budget, unqueried)

    def first_outcomes(self, n1: int, k1: int, n2: int, k2: int) -> list[tuple[float, State]]:
        a1, a2, p = 2.0 ** (k1 - n1), 2.0 ** (k2 - n2), self.prior
        candidates = [
            (p * a2 + (1 - p) * a1, ((n1, k1, 1), (n2, k2, 1))),
            (p * (1 - a2), ((n1, k1, 1), (n2, 0, 0))),
            ((1 - p) * (1 - a1), ((n1, 0, 0), (n2, k2, 1))),
        ]
        return [(mass, s) for mass, s in candidates if mass > 0]

    def first_value(self, plan: tuple[int, int, int, int], budget: int) -> float:
        return sum(prob * self.fixed_continuation(s, budget).population_risk
                   for prob, s in self.first_outcomes(*plan))

    def two_batch_profile(self, budget: int) -> list[dict]:
        """Full two-batch optimum at EVERY first-batch size, including empty batches."""
        self._validate_budget(budget)
        profile = []
        for t in range(budget + 1):
            best, best_plan = np.inf, None
            for n1 in range(max(0, t - self.m), min(t, self.m) + 1):
                n2 = t - n1
                for k1 in self.feasible_ranks(n1):
                    for k2 in self.feasible_ranks(n2):
                        plan = (n1, k1, n2, k2)
                        value = self.first_value(plan, budget)
                        if value < best:
                            best, best_plan = value, plan
            profile.append(dict(first_batch=t, population_risk=float(best), first_plan=best_plan))
        return profile

    def two_batch(self, budget: int, unqueried: bool = True) -> float:
        value = min(row["population_risk"] for row in self.two_batch_profile(budget))
        return self.normalize(value, budget, unqueried)

    def subcube(self, budget: int, unqueried: bool = True) -> SubcubeValue:
        """Definition 1: empty, region-one cube, or equal cubes in both regions."""
        self._validate_budget(budget)
        plans = [(0, 0, 0, 0)]
        for s in range(2, min(self.d, 5) + 1):
            t, k = 1 << s, s + 1
            if t <= budget:
                plans.append((t, k, 0, 0))
            if 2 * t <= budget:
                plans.append((t, k, t, k))
        value, plan = min((self.first_value(plan, budget), plan) for plan in plans)
        return SubcubeValue(self.normalize(value, budget, unqueried), plan)

    def _successors(self, state: State, j: int):
        n, k, o = state[j]
        if n == self.m:
            return
        def changed(region):
            s = list(state)
            s[j] = region
            return tuple(s)
        if not o:
            yield "inactive", self.m - n, [(1.0, changed((n + 1, 0, 0)))]
            return
        if k < self.r:
            yield "expand", self.m - span_size(k), [(1.0, changed((n + 1, k + 1, 1)))]
        if span_size(k) > n:
            gamma = self.posterior(state)
            gamma = gamma if j == 0 else 1 - gamma
            outcomes = [((1 + gamma) / 2, changed((n + 1, k, 1)))]
            if gamma < 1:
                outcomes.append(((1 - gamma) / 2, changed((n + 1, 0, 0))))
            yield "check", span_size(k) - n, outcomes

    def adaptive(self, budget: int, unqueried: bool = True) -> float:
        self._validate_budget(budget)
        @lru_cache(maxsize=None)
        def solve(state: State) -> float:
            if sum(s[0] for s in state) == budget:
                return self.terminal_loss(state)
            return min(sum(prob * solve(next_state) for prob, next_state in outcomes)
                       for j in range(2) for _, _, outcomes in self._successors(state, j))
        return self.normalize(solve(EMPTY), budget, unqueried)

    def uniform_recursive(self, budget: int, unqueried: bool = True) -> float:
        """Independent validation of Uniform via the identity-count recursion (151)."""
        self._validate_budget(budget)
        @lru_cache(maxsize=None)
        def solve(state: State) -> float:
            count = sum(s[0] for s in state)
            if count == budget:
                return self.terminal_loss(state)
            return sum(multiplicity * sum(prob * solve(next_state) for prob, next_state in outcomes)
                       for j in range(2) for _, multiplicity, outcomes in self._successors(state, j)
                       ) / (2 * self.m - count)
        return self.normalize(solve(EMPTY), budget, unqueried)

    def subset_rank_counts(self, max_count: int) -> list[dict[int, int]]:
        """Exact integer recurrence (152), including the empty rank-zero subset."""
        counts = [{0: 1}]
        for n in range(1, min(max_count, self.m) + 1):
            row = {}
            for k in self.feasible_ranks(n):
                numerator = (counts[n - 1].get(k, 0) * (span_size(k) - n + 1)
                             + counts[n - 1].get(k - 1, 0) * (self.m - span_size(k - 1)))
                value, remainder = divmod(numerator, n)
                if remainder:
                    raise ArithmeticError("Subset-count recurrence lost integrality")
                row[k] = value
            if sum(row.values()) != comb(self.m, n):
                raise ArithmeticError("Subset-rank counts do not normalize")
            counts.append(row)
        return counts

    def uniform(self, budget: int, unqueried: bool = True) -> float:
        self._validate_budget(budget)
        counts, denominator = self.subset_rank_counts(budget), comb(2 * self.m, budget)
        total = 0.0
        for n1 in range(max(0, budget - self.m), min(budget, self.m) + 1):
            n2 = budget - n1
            for k1, c1 in counts[n1].items():
                for k2, c2 in counts[n2].items():
                    # Fixed-set integrated terminal cost is the no-continuation case.
                    risk = sum(prob * self.terminal_loss(s)
                               for prob, s in self.first_outcomes(n1, k1, n2, k2))
                    total += (c1 * c2 / denominator) * risk
        return self.normalize(total, budget, unqueried)


def basis_risk(d: int, budget: int, eta: float = 0.0, prior: float = 0.5,
               unqueried: bool = True) -> float:
    """Theorem 6: one exact value for all three acquisition classes, Eq. (138)."""
    m, r = 1 << d, d + 1
    if not 0 <= budget <= 2 * r or not 0 <= eta < 0.5 or not 0 <= prior <= 1:
        raise ValueError("Invalid basis-pool experiment")
    delta = 1 - 2 * eta
    h = [sum(comb(n, k) * delta ** (2 * k + 2) for k in range(3, n + 1, 2))
         for n in range(r + 1)]
    gain = max(prior ** 2 * h[n] + (1 - prior) ** 2 * h[budget - n]
               for n in range(max(0, budget - r), min(r, budget) + 1))
    risk = 0.25 - (budget + gain) / (8 * m)
    if unqueried:
        if budget == 2 * m:
            raise ValueError("No unqueried targets")
        risk *= 2 * m / (2 * m - budget)
    return risk
