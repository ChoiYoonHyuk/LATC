from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.stats import binom
from .affine import augmented_rank, validate_noise
from .paired import write_csv

RELATION_PROBES = {
    "length_four": (0, 1, 2, 3, 4, 8, 16, 32),
    "length_eight": (0, 1, 2, 4, 8, 16, 32, 63),
    "independent": (0, 1, 2, 4, 8, 16, 32, 64),
    "subcube_eight": tuple(range(8)),
}


def dependency_code(coordinates):
    rows = [(int(x) << 1) | 1 for x in coordinates]
    result = []
    for mask in range(1 << len(rows)):
        total = 0
        for j, row in enumerate(rows):
            if mask & (1 << j):
                total ^= row
        if total == 0:
            result.append(mask)
    return tuple(result)


def likelihood_fourier(coordinates, labels, eta: float):
    validate_noise(eta)
    labels = tuple(int(y) for y in labels)
    if len(labels) != len(coordinates) or any(y not in (0, 1) for y in labels):
        raise ValueError("invalid observed labels")
    observed = sum(y << j for j, y in enumerate(labels))
    delta = 1-2*eta
    return sum(delta**mask.bit_count() * (-1 if (mask & observed).bit_count() & 1 else 1)
               for mask in dependency_code(coordinates))


def relation_statistics(coordinates, eta: float):
    validate_noise(eta)
    n = len(coordinates)
    code = dependency_code(coordinates)
    delta = 1-2*eta
    likelihoods = np.array([sum(delta**v.bit_count() * (-1 if (v & y).bit_count() & 1 else 1)
                                 for v in code) for y in range(1 << n)])

    error = float(np.minimum(1.0, likelihoods).mean()/2)
    chi2 = sum(delta**(2*v.bit_count()) for v in code if v)
    return {"count": n, "rank": augmented_rank(coordinates),
            "dependency_weights": [v.bit_count() for v in code if v],
            "testing_error": error, "chi_squared": chi2,
            "total_variation": float(np.mean(np.abs(likelihoods-1))/2)}


def binomial_diagnostic(L: int, normalized_noise: float):

    if L < 2 or normalized_noise < 0 or not np.isfinite(normalized_noise):
        raise ValueError("L>=2 and nonnegative finite normalized noise are required")
    r = 1 << L
    k = (2*r + L-1)//L
    s = math.isqrt(L)
    rho = L*math.log(L)/r
    eta = normalized_noise*rho
    if eta >= 0.5:
        return None
    signal = math.exp((2*k)*math.log1p(-2*eta)) if eta else 1.0
    successes = np.arange(s+1)
    reliable = binom.pmf(successes, s, (1+signal)/2)
    independent = binom.pmf(successes, s, 0.5)
    error = float(0.5*np.minimum(reliable, independent).sum())
    return {"L": L, "r": str(r), "rho": rho, "normalized_noise": normalized_noise,
            "eta": eta, "length_cap": str(2*k), "checks": s,
            "probe_cap": str(2*k*s), "probe_cap_over_r": (2*k*s)/r,
            "parity_signal": signal, "testing_error": error}


def terminal_components(eta: float, dependency_length: int):

    validate_noise(eta)
    if dependency_length not in (4, 6):
        raise ValueError("dependency length must be 4 or 6")
    delta = 1-2*eta
    k = dependency_length
    if eta == 0:

        F, G = 10.0, 7.5
    else:
        u = delta**k
        A = 16*delta**6+4*delta**10 if k == 4 else 20*delta**6
        B = 12*delta**6+8*delta**8 if k == 4 else 20*delta**6
        F = (A-B*u)/(1-u*u)
        G = (A+F)/4
    residual = 0.25-delta**2/8
    task = delta**2*(10-F)/80
    reliability = delta**2*(F-G)/80
    return {"eta": eta, "dependency_length": k, "F": F, "G": G,
            "irreducible": residual, "task_uncertainty": task,
            "reliability_uncertainty": reliability,
            "risk_unqueried": residual+task+reliability}


def terminal_gap(eta: float):
    z = (1-2*eta)**2
    polynomial = z**6+z**5+z**4+4*z**3+5*z**2+2*z+2
    return z**4*(1-z)*polynomial/(80*(1+z*z)*(1+z**3))


def run_mechanisms(out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    relation_rows = []

    for eta in sorted(set(np.linspace(0, 0.49, 99).tolist()+[0.05, 0.10, 0.25])):
        for name, coordinates in RELATION_PROBES.items():
            relation_rows.append({"probe": name, "eta": float(eta), **relation_statistics(coordinates, float(eta))})
    write_csv(out/"rq4_relations.csv", relation_rows)
    diagnostics = []
    for L in (6, 16, 64, 256):
        for normalized in sorted(set(np.geomspace(1e-3, 2, 201).tolist()+[0.0, 1.0])):
            row = binomial_diagnostic(L, float(normalized))
            if row is not None:
                diagnostics.append(row)
    write_csv(out/"rq4_noise_scale.csv", diagnostics)
    overhead = []
    for r in (64, 256, 1024, 4096, 16384, 65536):
        L = r.bit_length()-1
        k = (2*r+L-1)//L
        s = math.isqrt(L)
        overhead.append({"r": r, "length_cap": 2*k, "checks": s,
                         "probe_cap": 2*k*s, "probe_cap_over_r": 2*k*s/r})
    write_csv(out/"rq4_overhead.csv", overhead)
    components = [terminal_components(eta, k) for eta in (0.0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.49) for k in (4, 6)]
    write_csv(out/"rq3_terminal_components.csv", components)
    return {"relation_rows": len(relation_rows), "noise_diagnostic_rows": len(diagnostics),
            "diagnostic_is_analytic_not_large_population_simulation": True}
