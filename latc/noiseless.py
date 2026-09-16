from __future__ import annotations
from functools import lru_cache
from itertools import product
import math
from .affine import validate_dimension, validate_noise

Region = tuple[int, int, bool]
State = tuple[Region, Region]
EMPTY: State = ((0, 0, True), (0, 0, True))


def closure_size(rank: int) -> int:
    return 0 if rank == 0 else 1 << (rank - 1)


def replace_region(state: State, region: int, value: Region) -> State:
    return (value, state[1]) if region == 0 else (state[0], value)


class NoiselessSolver:

    def __init__(self, d: int, prior: float = 0.5):
        validate_dimension(d)
        if not 0 <= prior <= 1:
            raise ValueError("prior must be in [0, 1]")
        self.d, self.r, self.m, self.prior = d, d + 1, 1 << d, prior

        self.gamma = lru_cache(None)(self._gamma)
        self.terminal = lru_cache(None)(self._terminal)
        self.continuation_outcomes = lru_cache(None)(self._continuation_outcomes)
        self.best_fixed = lru_cache(None)(self._best_fixed)
        self.adaptive = lru_cache(None)(self._adaptive)
        self.uniform = lru_cache(None)(self._uniform)

    def clear_caches(self):
        for method in (self.gamma, self.terminal, self.continuation_outcomes,
                       self.best_fixed, self.adaptive, self.uniform):
            method.cache_clear()

    def _gamma(self, state: State) -> tuple[float, float]:
        likelihood = [2.0 ** (n - k) if ok else 0.0 for n, k, ok in state]
        weights = [self.prior * likelihood[0], (1 - self.prior) * likelihood[1]]
        z = sum(weights)
        if z <= 0:
            raise ValueError("impossible reduced state")
        return weights[0] / z, weights[1] / z

    def _terminal(self, state: State) -> float:
        gamma = self.gamma(state)
        return sum(self.m - n - (gamma[j] ** 2 * (closure_size(k) - n) if ok else 0)
                   for j, (n, k, ok) in enumerate(state)) / (8 * self.m)

    def options(self, region: Region, q: int):
        n, k, ok = region
        if q < 0 or n + q > self.m:
            return ()
        if not ok:
            return (0,)
        return tuple(a for a in range(min(q, self.r - k) + 1)
                     if n + q <= closure_size(k + a))

    def _continuation_outcomes(self, state: State, qs: tuple[int, int],
                               additions: tuple[int, int]):

        gamma = self.gamma(state)
        aggregate: dict[State, float] = {}
        for reliable in (0, 1):
            if gamma[reliable] == 0:
                continue
            regional = []
            for j, ((n, k, ok), q, a) in enumerate(zip(state, qs, additions)):
                if not ok:
                    regional.append((((n + q, 0, False), 1.0),))
                else:
                    success = 1.0 if j == reliable else 2.0 ** (a - q)
                    outcomes = [((n + q, k + a, True), success)]
                    if success < 1:
                        outcomes.append(((n + q, 0, False), 1 - success))
                    regional.append(tuple(outcomes))
            for (first, p0), (second, p1) in product(*regional):
                probability = gamma[reliable] * p0 * p1
                if probability:
                    nxt = (first, second)
                    aggregate[nxt] = aggregate.get(nxt, 0.0) + probability
        return tuple(aggregate.items())

    def _best_fixed(self, state: State, remaining: int):

        if remaining == 0:
            return self.terminal(state), ((0, 0), (0, 0))
        best = math.inf
        choice = None
        q0_min = max(0, remaining - (self.m - state[1][0]))
        q0_max = min(remaining, self.m - state[0][0])
        for q0 in range(q0_min, q0_max + 1):
            qs = (q0, remaining - q0)
            for a0, a1 in product(self.options(state[0], qs[0]), self.options(state[1], qs[1])):
                additions = (a0, a1)
                value = sum(prob * self.terminal(nxt)
                            for nxt, prob in self.continuation_outcomes(state, qs, additions))
                if value < best:
                    best, choice = value, (qs, additions)
        if choice is None:
            raise ValueError("infeasible fixed continuation budget")
        return best, choice

    def actions(self, state: State):
        gamma = self.gamma(state)
        for j, (n, k, ok) in enumerate(state):
            if n == self.m:
                continue
            if not ok:
                yield (j, "fill"), ((replace_region(state, j, (n + 1, 0, False)), 1.0),)
            else:
                if k < self.r:
                    yield (j, "expand"), ((replace_region(state, j, (n + 1, k + 1, True)), 1.0),)
                if closure_size(k) > n:
                    agree = (1 + gamma[j]) / 2
                    nxt = [(replace_region(state, j, (n + 1, k, True)), agree)]
                    if agree < 1:
                        nxt.append((replace_region(state, j, (n + 1, 0, False)), 1 - agree))
                    yield (j, "check"), tuple(nxt)

    def _adaptive(self, state: State, remaining: int):
        if remaining == 0:
            return self.terminal(state), None
        best, choice = math.inf, None
        for action, outcomes in self.actions(state):
            value = sum(prob * self.adaptive(nxt, remaining - 1)[0] for nxt, prob in outcomes)
            if value < best:
                best, choice = value, action
        if choice is None:
            raise ValueError("infeasible adaptive budget")
        return best, choice

    def _uniform(self, state: State, remaining: int) -> float:
        if remaining == 0:
            return self.terminal(state)
        total = 2 * self.m - sum(region[0] for region in state)
        answer = 0.0
        for (j, kind), outcomes in self.actions(state):
            n, k, ok = state[j]
            count = (self.m - n if kind == "fill" else
                     self.m - closure_size(k) if kind == "expand" else closure_size(k) - n)
            answer += count / total * sum(prob * self.uniform(nxt, remaining - 1)
                                          for nxt, prob in outcomes)
        return answer

    def latc(self, budget: int):

        value, _ = self.best_fixed(EMPTY, budget)
        best = (value, {"probe": "empty", "h": 0, "labels": 0})
        for h in range(2, min(self.d, 5) + 1):
            t = 1 << h
            for number in (1, 2):
                if number * t > budget:
                    continue
                qs = (t, 0) if number == 1 else (t, t)
                additions = (h + 1, 0) if number == 1 else (h + 1, h + 1)
                outcome_value = sum(prob * self.best_fixed(nxt, budget - number * t)[0]
                                    for nxt, prob in self.continuation_outcomes(EMPTY, qs, additions))
                if outcome_value < best[0]:
                    best = (outcome_value, {"probe": "region_one" if number == 1 else "both_regions",
                                            "h": h, "labels": number * t})
        return best

    def solve_budget(self, budget: int) -> dict:
        if not 0 <= budget < 2 * self.m:
            raise ValueError("budget must leave at least one unqueried target")
        fixed, allocation = self.best_fixed(EMPTY, budget)
        adaptive, _ = self.adaptive(EMPTY, budget)
        uniform = self.uniform(EMPTY, budget)
        latc, probe = self.latc(budget)
        basis, basis_allocation = basis_optimum(self.d, budget, 0.0, self.prior) if budget <= 2*self.r else (math.nan, None)
        factor = 2 * self.m / (2 * self.m - budget)
        return {"d": self.d, "r": self.r, "budget": budget, "prior": self.prior,
                "uniform_full": uniform, "fixed_full": fixed, "latc_full": latc,
                "adaptive_full": adaptive, "basis_full": basis,
                "uniform_unqueried": uniform * factor, "fixed_unqueried": fixed * factor,
                "latc_unqueried": latc * factor, "adaptive_unqueried": adaptive * factor,
                "basis_unqueried": basis * factor, "probe": probe["probe"],
                "probe_dimension": probe["h"], "probe_labels": probe["labels"],
                "fixed_counts": list(allocation[0]), "fixed_ranks": list(allocation[1]),
                "basis_counts": basis_allocation, "adaptive_cache_states": self.adaptive.cache_info().currsize}


def basis_optimum(d: int, budget: int, eta: float = 0.0, prior: float = 0.5):

    validate_dimension(d)
    validate_noise(eta)
    if not 0 <= prior <= 1:
        raise ValueError("prior must be in [0,1]")
    r, m = d + 1, 1 << d
    if not 0 <= budget <= 2*r:
        raise ValueError("basis budget must lie in [0, 2(d+1)]")
    delta = 1 - 2*eta
    h = [sum(math.comb(n, k)*delta**(2*k+2) for k in range(3, n+1, 2)) for n in range(r+1)]
    best, choice = math.inf, None
    for n0 in range(max(0, budget-r), min(r, budget)+1):
        n1 = budget-n0
        value = 0.25 - (budget + prior**2*h[n0] + (1-prior)**2*h[n1])/(8*m)
        if value < best:
            best, choice = value, [n0, n1]
    return best, choice
