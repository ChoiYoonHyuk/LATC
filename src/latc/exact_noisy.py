"""Complete finite-pool acquisition optimization, Appendix N.4.

Histories are base-three integers: 0=unobserved, 1=observed zero, 2=observed one.
The global index is h1 + 3**q1 * h2. Costs include P(H), so child costs SUM;
applying conditional probabilities again would be a double-weighting error.

Fixed continuation enumerates every final query subset. For each subset, a
ternary tensor contains all outcome marginals, and elementwise minima produce
F_b(H). This is exactly Eq. (194), not a greedy batch approximation.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.special import xlogy
from .posterior import regional_posterior

SHORT_POOL = (0, 1, 2, 3, 4, 8)
LONG_POOL = (0, 1, 2, 4, 8, 15)
COMMON_TARGETS = (5, 6, 7, 9, 10, 11, 12, 13, 14)


def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    return -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / np.log(2)


class ExactNoisyOptimizer:
    def __init__(self, d: int, pools, eta: float, prior: float = 0.5, targets=None, max_queries: int = 12):
        if not isinstance(d, int) or not 0 <= d <= 22:
            raise ValueError("Require integer 0 <= d <= 22")
        if not 0 <= prior <= 1 or not 0 <= eta < 0.5:
            raise ValueError("Invalid prior/noise")
        self.d, self.m, self.eta, self.prior = d, 1 << d, eta, prior
        raw_pools = tuple(np.asarray(p) for p in pools)
        self.pools = tuple(p.astype(np.int64) for p in raw_pools)
        if any(np.any(a != b) for a, b in zip(raw_pools, self.pools)):
            raise ValueError("Query coordinates must be integral")
        if len(self.pools) != 2:
            raise ValueError("Exactly two regional pools are required")
        self.sizes = tuple(len(p) for p in self.pools)
        self.K = sum(self.sizes)
        if sum(3**q for q in self.sizes) * self.m > 25_000_000:
            raise ValueError("Regional posterior tables exceed the 25-million-cell memory guard")
        if self.K > max_queries or self.K > 14:
            raise ValueError("Exhaustive noisy acquisition guard exceeded (default K <=12)")
        for p in self.pools:
            if len(np.unique(p)) != len(p) or np.any(p < 0) or np.any(p >= self.m):
                raise ValueError("Invalid pool coordinates")
        if targets is None:
            targets = tuple(np.setdiff1d(np.arange(self.m), p) for p in self.pools)
        self.targets = tuple(np.asarray(t, dtype=np.int64) for t in targets)
        if len(self.targets) != 2 or sum(map(len, self.targets)) == 0:
            raise ValueError("Two common target sets with nonzero total size are required")
        for t, p in zip(self.targets, self.pools):
            if len(np.unique(t)) != len(t) or np.any(t < 0) or np.any(t >= self.m) or np.intersect1d(t, p).size:
                raise ValueError("Common targets must be distinct, valid, and outside their query pool")
        self.powers = 3 ** np.arange(self.K, dtype=np.int64)
        self.nstates = 3 ** self.K
        self.states = np.arange(self.nstates, dtype=np.int64)
        self.digits = ((self.states[None, :] // self.powers[:, None]) % 3).astype(np.uint8)
        self.count = np.count_nonzero(self.digits, axis=0)
        self.by_count = [self.states[self.count == b] for b in range(self.K + 1)]
        self.local = [self._regional_table(j) for j in range(2)]
        a, b = self.local
        den = prior * a["L"][:, None] + (1 - prior) * b["L"][None, :]
        count = a["count"][:, None] + b["count"][None, :]
        probability = den * np.exp2(-count.astype(float))
        g1 = np.divide(prior * a["L"][:, None], den, out=np.zeros_like(den), where=den > 0)
        g2 = np.divide((1 - prior) * b["L"][None, :], den, out=np.zeros_like(den), where=den > 0)
        self.probability = probability.ravel(order="F")
        self.gamma1 = g1.ravel(order="F")
        delta2 = (1 - 2 * eta) ** 2
        common = 0.25 - delta2 * (g1 ** 2 * a["common_energy"][:, None]
                                 + g2 ** 2 * b["common_energy"][None, :]) / (4 * sum(map(len, self.targets)))
        population = (2 * self.m - count - delta2 * (g1 ** 2 * a["unqueried_energy"][:, None]
                       + g2 ** 2 * b["unqueried_energy"][None, :])) / (8 * self.m)
        self.costs = {"common": (probability * common).ravel(order="F"),
                      "population": (probability * population).ravel(order="F")}
        self._common_entropy = None
        # Keeping only a modest number of subset index arrays bounds cache memory.
        self.complete_codes = lru_cache(maxsize=256)(self._complete_codes)
        self._greedy_batch_cache = {}

    def _regional_table(self, region: int) -> dict:
        pool, size = self.pools[region], self.sizes[region]
        states = 3 ** size
        out = {"L": np.empty(states), "count": np.empty(states, dtype=np.int16),
               "mean": np.empty((states, self.m)), "common_energy": np.empty(states),
               "unqueried_energy": np.empty(states)}
        for code in range(states):
            digits = (code // (3 ** np.arange(size))) % 3
            seen = digits > 0
            q, y = pool[seen], digits[seen] - 1
            posterior = regional_posterior(self.d, q, y, self.eta)
            means = posterior.sign_mean
            out["L"][code] = np.exp(posterior.log_likelihood_ratio)
            out["count"][code] = seen.sum()
            out["mean"][code] = means
            out["common_energy"][code] = np.sum(means[self.targets[region]] ** 2)
            unqueried = np.ones(self.m, dtype=bool)
            unqueried[q] = False
            out["unqueried_energy"][code] = np.sum(means[unqueried] ** 2)
        return out

    def _complete_codes(self, subset: tuple[int, ...]) -> np.ndarray:
        # All coordinates in subset are observed, with all 2^|subset| outcomes.
        codes = np.array([0], dtype=np.int64)
        for i in subset:
            codes = np.concatenate((codes + self.powers[i], codes + 2 * self.powers[i]))
        return codes

    def partial_codes(self, subset: tuple[int, ...]) -> np.ndarray:
        codes = np.array([0], dtype=np.int64)
        for i in subset:
            codes = np.concatenate((codes, codes + self.powers[i], codes + 2 * self.powers[i]))
        return codes

    def _validate(self, budget: int, metric: str):
        if not isinstance(budget, int) or not 0 <= budget <= self.K:
            raise ValueError("Budget must be an integer in [0,K]")
        if metric not in self.costs:
            raise ValueError("Metric must be 'common' or 'population'")

    def adaptive(self, budget: int, metric: str = "common") -> float:
        self._validate(budget, metric)
        value = self.costs[metric].copy()
        for n in range(budget - 1, -1, -1):
            states = self.by_count[n]
            value[states] = np.inf
            for v, power in enumerate(self.powers):
                h = states[self.digits[v, states] == 0]
                candidate = value[h + power] + value[h + 2 * power]
                value[h] = np.minimum(value[h], candidate)
        return float(value[0])

    def fixed_continuation_costs(self, budget: int, metric: str = "common") -> np.ndarray:
        self._validate(budget, metric)
        cost = self.costs[metric]
        best = np.full(self.nstates, np.inf)
        for subset in combinations(range(self.K), budget):
            marginals = cost[self.complete_codes(subset)].reshape((2,) * budget, order="F")
            for axis in range(budget):
                # Axis index 0 now means 'unobserved', and sums both labels.
                marginals = np.concatenate((marginals.sum(axis=axis, keepdims=True), marginals), axis=axis)
            h = self.partial_codes(subset)
            best[h] = np.minimum(best[h], marginals.ravel(order="F"))
        return best

    def optimize(self, budget: int, metric: str = "common") -> dict[str, float]:
        continuation = self.fixed_continuation_costs(budget, metric)
        two_batch = np.inf
        for n in range(budget + 1):
            for subset in combinations(range(self.K), n):
                two_batch = min(two_batch, float(continuation[self.complete_codes(subset)].sum()))
        result = {"fixed": float(continuation[0]), "two_batch": float(two_batch),
                  "adaptive": self.adaptive(budget, metric)}
        if result["adaptive"] > result["two_batch"] + 2e-11 or result["two_batch"] > result["fixed"] + 2e-11:
            raise ArithmeticError("Acquisition-class ordering failed")
        return result

    def common_entropy_cost(self) -> np.ndarray:
        """History-weighted mean binary entropy of the same fixed common targets."""
        if self._common_entropy is None:
            n1, n2 = 3 ** self.sizes[0], 3 ** self.sizes[1]
            gamma1 = self.gamma1.reshape(n1, n2, order="F")
            prob = self.probability.reshape(n1, n2, order="F")
            entropy = np.zeros((n1, n2))
            for j, targets in enumerate(self.targets):
                gamma = gamma1 if j == 0 else 1 - gamma1
                for x in targets:
                    means = self.local[j]["mean"][:, x]
                    means = means[:, None] if j == 0 else means[None, :]
                    p = 0.5 - (1 - 2 * self.eta) * gamma * means / 2
                    entropy += binary_entropy(p)
            self._common_entropy = (prob * entropy / sum(map(len, self.targets))).ravel(order="F")
        return self._common_entropy

    def sequential_objective(self, objective: str, tie: str = "public", tolerance: float = 1e-12
                             ) -> list[float]:
        """Exact integration of BALD/EPIG/one-step Brier at all budgets (O.1--O.2)."""
        if objective not in ("bald", "epig", "brier"):
            raise ValueError("Unknown acquisition objective")
        if tie not in ("public", "reverse", "interleaved", "uniform"):
            raise ValueError("Unknown acquisition-score tie convention")
        q1, q2 = self.sizes
        if tie == "reverse":
            order = list(range(q1 - 1, -1, -1)) + list(range(self.K - 1, q1 - 1, -1))
        elif tie == "interleaved":
            order = []
            for i in range(max(q1, q2)):
                if i < q1: order.append(i)
                if i < q2: order.append(q1 + i)
        else:
            order = list(range(self.K))
        criterion = (self.common_entropy_cost() if objective == "epig" else self.costs["common"])
        mass = np.zeros(self.nstates)
        mass[0] = 1
        risks = []
        for n in range(self.K + 1):
            h = self.by_count[n]
            h = h[mass[h] > 0]
            ph = self.probability[h]
            risks.append(float(np.sum(mass[h] * self.costs["common"][h] / ph)))
            if n == self.K:
                break
            scores = np.full((self.K, len(h)), -np.inf)
            for v, power in enumerate(self.powers):
                valid = self.digits[v, h] == 0
                s = h[valid]
                if objective == "bald":
                    p1 = self.probability[s + 2 * power] / self.probability[s]
                    gamma = self.gamma1[s] if v < q1 else 1 - self.gamma1[s]
                    scores[v, valid] = binary_entropy(p1) - gamma * binary_entropy(self.eta) - (1 - gamma)
                else:
                    scores[v, valid] = (criterion[s] - criterion[s + power] - criterion[s + 2 * power]) / self.probability[s]
            maximum = scores.max(axis=0)
            tied = scores >= maximum[None, :] - tolerance
            if tie == "uniform":
                action_weights = tied / tied.sum(axis=0)[None, :]
            else:
                choice = np.asarray(order)[np.argmax(tied[order], axis=0)]
                action_weights = np.zeros_like(scores)
                action_weights[choice, np.arange(len(h))] = 1
            for v, power in enumerate(self.powers):
                active = action_weights[v] > 0
                s, weights = h[active], action_weights[v, active]
                for digit in (1, 2):
                    child = s + digit * power
                    contribution = mass[s] * weights * self.probability[child] / self.probability[s]
                    np.add.at(mass, child, contribution)
            if abs(mass[self.by_count[n + 1]].sum() - 1) > 1e-9:
                raise ArithmeticError("Policy history probabilities failed to normalize")
        return risks

    def _greedy_batch(self, history: int, size: int, tolerance: float = 1e-12) -> tuple[int, ...]:
        key = (history, size, tolerance)
        if key in self._greedy_batch_cache:
            return self._greedy_batch_cache[key]
        available = [v for v in range(self.K) if self.digits[v, history] == 0]
        selected: list[int] = []
        ph = self.probability[history]
        if ph <= 0 or size > len(available):
            raise ValueError("Impossible BatchBALD state/batch")
        gamma = self.gamma1[history]
        conditional_entropy = [((gamma if v < self.sizes[0] else 1 - gamma)
                                 * float(binary_entropy(self.eta))
                                 + (1 - (gamma if v < self.sizes[0] else 1 - gamma)))
                               for v in range(self.K)]
        for _ in range(size):
            best_score, best_v = -np.inf, None
            for v in available:
                if v in selected:
                    continue
                subset = tuple(sorted(selected + [v]))
                probs = self.probability[history + self.complete_codes(subset)] / ph
                entropy = -float(np.sum(xlogy(probs, probs))) / np.log(2)
                score = entropy - sum(conditional_entropy[i] for i in subset)
                if best_v is None or score > best_score + tolerance:
                    best_score, best_v = score, v
            selected.append(best_v)
        result = tuple(sorted(selected))
        self._greedy_batch_cache[key] = result
        return result

    def batchbald(self, budget: int) -> float:
        """Greedy joint-information batches with first size floor(b/2), as in O.1."""
        self._validate(budget, "common")
        first = self._greedy_batch(0, budget // 2)
        total = 0.0
        for h in self.complete_codes(first):
            if self.probability[h] == 0:
                continue
            second = self._greedy_batch(int(h), budget - len(first))
            total += float(self.costs["common"][h + self.complete_codes(second)].sum())
        return total
