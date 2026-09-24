"""Array-level paired inference; no treating budgets/policies as independent data."""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t, beta


def student_summary(values, alpha: float = 0.05, family_size: int = 1) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite independent sampling-unit values are required")
    if not 0 < alpha < 1 or family_size < 1:
        raise ValueError("Invalid confidence level/multiplicity")
    n = len(x)
    mean, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(1 - alpha / (2 * family_size), n - 1))
    width = critical * se
    pvalue = float(2 * student_t.sf(abs(mean / se), n - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    return dict(n=n, mean=mean, standard_error=se, ci_lower=mean-width, ci_upper=mean+width,
                half_width=width, p_value=pvalue, degrees_of_freedom=n-1,
                alpha=alpha, family_size=family_size)


def paired_summary(left, right, **kwargs) -> dict:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired arrays must have matching shapes and order")
    return student_summary(x - y, **kwargs)


def holm_adjust(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or np.any(p > 1) or not np.isfinite(p).all():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def clopper_pearson(k: int, n: int, alpha: float = 0.025) -> tuple[float, float]:
    """Each model gets coverage 1-alpha=.975 in the two-model test experiment."""
    if not 0 <= k <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count/confidence level")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha/2, k, n-k+1))
    upper = 1.0 if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))
    return lower, upper


def equal_prior_error_interval(k_affine: int, k_independent: int, n: int) -> dict:
    lo1, hi1 = clopper_pearson(k_affine, n)
    lo0, hi0 = clopper_pearson(k_independent, n)
    return dict(affine_rejections=k_affine, independent_acceptances=k_independent,
                samples_per_model=n, estimated_error=(k_affine+k_independent)/(2*n),
                ci_lower=(lo1+lo0)/2, ci_upper=(hi1+hi0)/2,
                interval="two 97.5% Clopper-Pearson intervals; union-bound 95% combined coverage")
