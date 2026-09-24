"""Exact Bayes inference: Eqs. (4), (159)--(161), and (177).

Only inference is exponential in d; no full 2^(2m) label-array enumeration is
used.  Queried labels are persistent observations, not fresh noisy responses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
from scipy.special import logsumexp
from .algebra import fwht


@dataclass(frozen=True)
class RegionalPosterior:
    d: int
    eta: float
    coordinates: np.ndarray
    labels: np.ndarray
    log_likelihood_ratio: float
    parameter_probabilities: np.ndarray  # shape (2,m), indexed [beta,u]
    sign_mean: np.ndarray


def validate_observations(d: int, coordinates: Sequence[int], labels: Sequence[int], eta: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or not 0 <= d <= 22:
        raise ValueError("Require an integer 0 <= d <= 22 (explicit population guard)")
    if not math.isfinite(eta) or not 0 <= eta < 0.5:
        raise ValueError("Require 0 <= eta < 1/2")
    rawq, rawy = np.asarray(coordinates), np.asarray(labels)
    if rawq.ndim != 1 or rawy.ndim != 1 or rawq.size != rawy.size:
        raise ValueError("Coordinates and labels must be equal-length vectors")
    q, y = rawq.astype(np.int64), rawy.astype(np.int8)
    if np.any(rawq != q) or np.any(rawy != y):
        raise ValueError("Coordinates and labels must be integral")
    if np.any(q < 0) or np.any(q >= 1 << d) or np.unique(q).size != q.size:
        raise ValueError("Coordinates must be distinct and within the regional cube")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    return q, y


def regional_posterior(d: int, coordinates: Sequence[int], labels: Sequence[int], eta: float
                       ) -> RegionalPosterior:
    q, y = validate_observations(d, coordinates, labels, eta)
    m, n = 1 << d, q.size
    observed_signs = np.zeros(m, dtype=np.int64)
    observed_signs[q] = 1 - 2 * y.astype(np.int64)
    correlation = fwht(observed_signs)
    disagreement = np.stack(((n - correlation) // 2, (n + correlation) // 2))
    if eta == 0:
        weights = (disagreement == 0).astype(float)
        mass = weights.sum()
        if mass == 0:
            return RegionalPosterior(d, eta, q, y, -math.inf, weights, np.zeros(m))
        posterior = weights / mass
        log_mass = math.log(mass)
    else:
        log_weights = (n - disagreement) * math.log1p(-eta) + disagreement * math.log(eta)
        log_mass = float(logsumexp(log_weights))
        posterior = np.exp(log_weights - log_mass)
    log_ratio = n * math.log(2) + log_mass - (d + 1) * math.log(2)
    # Independent rows have a mathematically exact likelihood of one. The
    # first-batch policy separately uses Eq. (3) so ties are never numerical drift.
    means = np.clip(fwht(posterior[0] - posterior[1]), -1.0, 1.0)
    return RegionalPosterior(d, eta, q, y, log_ratio, posterior, means)


@dataclass(frozen=True)
class Prediction:
    regional: tuple[RegionalPosterior, RegionalPosterior]
    gamma: np.ndarray
    probabilities: np.ndarray

    @property
    def population_risk(self) -> float:
        p = self.probabilities
        return float(np.mean(p * (1 - p)))

    @property
    def unqueried_risk(self) -> float:
        n = self.probabilities.size
        b = sum(len(r.coordinates) for r in self.regional)
        if b == n:
            raise ValueError("Unqueried risk is undefined when every target was acquired")
        return n / (n - b) * self.population_risk

    def common_risk(self, targets: Sequence[Sequence[int]]) -> float:
        return self.decomposition(targets)["conditional_brier"]

    def decomposition(self, targets: Sequence[Sequence[int]]) -> dict[str, float]:
        if len(targets) != 2:
            raise ValueError("Two regional evaluation sets are required")
        sizes, energies = [], []
        for r, t in zip(self.regional, targets):
            v = np.asarray(t, dtype=np.int64)
            if v.ndim != 1 or np.unique(v).size != v.size or np.any(v < 0) or np.any(v >= 1 << r.d):
                raise ValueError("Invalid evaluation coordinates")
            if np.intersect1d(v, r.coordinates).size:
                raise ValueError("Common targets must be unqueried")
            sizes.append(v.size)
            energies.append(float(np.sum(r.sign_mean[v] ** 2)))
        count = sum(sizes)
        if count == 0:
            raise ValueError("The common evaluation population must be nonempty")
        delta2 = (1 - 2 * self.regional[0].eta) ** 2
        sizes, energies = np.asarray(sizes), np.asarray(energies)
        scale = delta2 / (4 * count)
        residual = 0.25 - scale * float(self.gamma @ sizes)
        parameter = scale * float(self.gamma @ (sizes - energies))
        indicator = scale * float((self.gamma * (1 - self.gamma)) @ energies)
        risk = 0.25 - scale * float((self.gamma ** 2) @ energies)
        return dict(conditional_brier=risk, residual=residual,
                    parameter_uncertainty=parameter, indicator_uncertainty=indicator)

    def realized_risk(self, labels: np.ndarray, targets: Sequence[Sequence[int]] | None = None) -> float:
        labels = np.asarray(labels)
        if labels.shape != self.probabilities.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Invalid complete persistent label array")
        errors = (self.probabilities - labels) ** 2
        if targets is None:
            return float(errors.mean())
        parts = [errors[j, np.asarray(t, dtype=int)] for j, t in enumerate(targets)]
        if sum(v.size for v in parts) == 0:
            raise ValueError("Empty evaluation population")
        return float(sum(v.sum() for v in parts) / sum(v.size for v in parts))


def combine_regions(regional: tuple[RegionalPosterior, RegionalPosterior], prior: float = 0.5
                    ) -> Prediction:
    if not 0 <= prior <= 1 or not math.isfinite(prior):
        raise ValueError("The exclusive region-one prior must be in [0,1]")
    if regional[0].d != regional[1].d or regional[0].eta != regional[1].eta:
        raise ValueError("This implementation uses equal-sized regions and common noise")
    log_prior = [math.log(prior) if prior > 0 else -math.inf,
                 math.log1p(-prior) if prior < 1 else -math.inf]
    log_mass = np.array([log_prior[j] + regional[j].log_likelihood_ratio for j in range(2)])
    norm = float(logsumexp(log_mass))
    if not math.isfinite(norm):
        raise ValueError("The joint history has zero probability under the supplied model")
    gamma = np.exp(log_mass - norm)
    probabilities = np.stack([0.5 - gamma[j] * (1 - 2 * regional[j].eta)
                              * regional[j].sign_mean / 2 for j in range(2)])
    probabilities = np.clip(probabilities, 0, 1)
    for j, r in enumerate(regional):
        probabilities[j, r.coordinates] = r.labels
    return Prediction(regional, gamma, probabilities)


def predict_two_regions(d: int, coordinates: Sequence[Sequence[int]],
                        labels: Sequence[Sequence[int]], eta: float, prior: float = 0.5) -> Prediction:
    if len(coordinates) != 2 or len(labels) != 2:
        raise ValueError("Exactly two regional observation histories are required")
    return combine_regions(tuple(regional_posterior(d, coordinates[j], labels[j], eta)
                                 for j in range(2)), prior)
