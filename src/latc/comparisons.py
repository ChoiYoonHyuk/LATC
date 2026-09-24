"""Matched-pool full-acquisition prediction formulas (N.2--N.3)."""
from __future__ import annotations
import numpy as np
from .exact_noisy import ExactNoisyOptimizer, binary_entropy


def matched_common_brier(check_length: int, eta: float, prior: float = 0.5) -> float:
    """Eq. (193), including its continuous endpoint values."""
    if check_length not in (4, 6) or not 0 <= eta <= .5 or not 0 <= prior <= 1:
        raise ValueError("Invalid matched-pool parameters")
    if eta == 0:
        return 1/8 + prior*(1-prior)/8
    if eta == .5:
        return .25
    delta = 1 - 2*eta
    if check_length == 4:
        a, b = 15*delta**6 + 3*delta**10, 12*delta**6 + 6*delta**8
    else:
        a = b = 18*delta**6
    u, v = delta**check_length, (2*prior-1)**2
    f = (a-b*u)/(1-u*u)
    g = (1+v)*f/4 + ((1+v)*a - 2*v*b*u)/(4*(1-v*u*u))
    return .25 - delta*delta*g/72


def common_brier_gap(eta: float) -> float:
    """Positive equal-prior polynomial gap, Eq. (182)."""
    if not 0 <= eta <= .5:
        raise ValueError("Invalid noise")
    z = (1-2*eta)**2
    polynomial = z**6 + z**5 + z**4 + 4*z**3 + 6*z**2 + 2*z + 2
    return z**4 * (1-z) * polynomial / (96*(1+z*z)*(1+z**3))


def complete_target_scores(solver: ExactNoisyOptimizer) -> dict:
    """Direct sum over every full query outcome and every common target (Table 22)."""
    codes = solver.complete_codes(tuple(range(solver.K)))
    mass = solver.probability[codes]
    h1 = codes % (3**solver.sizes[0])
    h2 = codes // (3**solver.sizes[0])
    gamma1 = solver.gamma1[codes]
    totals = dict(brier=0.0, log_loss_bits=0.0, classification=0.0,
                  confidence_moment_1=0.0, confidence_moment_2=0.0,
                  confidence_moment_3=0.0, confidence_moment_4=0.0)
    count = sum(map(len, solver.targets))
    for j, targets in enumerate(solver.targets):
        gamma = gamma1 if j == 0 else 1-gamma1
        histories = h1 if j == 0 else h2
        for x in targets:
            means = solver.local[j]["mean"][histories, x]
            confidence = np.abs((1-2*solver.eta)*gamma*means)
            p = (1+confidence)/2
            totals["brier"] += float(mass @ (p*(1-p))) / count
            totals["log_loss_bits"] += float(mass @ binary_entropy(p)) / count
            totals["classification"] += float(mass @ ((1-confidence)/2)) / count
            for order in range(1,5):
                totals[f"confidence_moment_{order}"] += float(mass @ (confidence**order)) / count
    return totals
