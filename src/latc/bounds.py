"""Numerical bound helpers and explicitly certified finite inequalities.

Certificate decisions use directed-rounding Decimal intervals. Floating-point
optimization only proposes a valid scalar threshold; its optimality is NOT
needed for a certificate. The direct rows use Table 31's explicit cutoffs.
"""
from __future__ import annotations
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from math import comb, floor, log, log2
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import binom
from .intervals import Interval, IntervalArithmetic

# Explicit design choices, NOT precomputed output risks.
DIRECT_CUTOFFS = {
    768: dict(T=433, kappa=30, a=45, tau_test="0.1925", tau_dec="0.0882"),
    1024: dict(T=602, kappa=36, a=54, tau_test="0.1925", tau_dec="0.0897"),
    2048: dict(T=1242, kappa=64, a=99, tau_test="0.1925", tau_dec="0.0919"),
    4096: dict(T=2452, kappa=120, a=203, tau_test="0.2037", tau_dec="0.0927"),
}


def entropy(eta: float) -> float:
    if not 0 <= eta <= 1:
        raise ValueError("Probability must lie in [0,1]")
    return 0.0 if eta in (0, 1) else -(eta * log2(eta) + (1 - eta) * log2(1 - eta))


def subcube_size(r: int) -> int:
    if not isinstance(r, int) or r < 3:
        raise ValueError("The theorem-scale subcube rule requires integer r >=3")
    return 1 << math.ceil(log2(log(r) ** 2))


def information_spectrum(n: int, r: int, eta: float) -> tuple[float, int]:
    """Eq. (10): float evaluation over ALL binomial support points; returns scout a.

    This routine is for exploration. Directed-rounding certification evaluates
    the selected support point again with ``spectrum_interval`` below.
    """
    if n < 0 or r < 0 or not 0 <= eta < 0.5:
        raise ValueError("Invalid information-spectrum arguments")
    if eta == 0:
        return min(1.0, 2.0 ** min(n - r, 0)), 0
    a = np.arange(n + 1)
    exponent = n * log2(2 * (1 - eta)) + a * log2(eta / (1 - eta)) - r
    first = np.exp2(np.minimum(exponent, 1))
    tail = binom.cdf(a - 1, n, eta)  # STRICT event B_n < a.
    values = first + tail
    index = int(np.argmin(values))
    return min(1.0, float(values[index])), index


def _natural_entropy(p: Interval) -> Interval:
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _kl(p: Interval, q: Interval) -> Interval:
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def spectrum_interval(n: int, r: int, eta: Interval, a: int) -> Interval:
    """Likelihood split at S_n(a); Eq. (218) for eta=1/20.

    The binomial sum uses nonnegative recurrences with directed rounding. For
    each chosen a this is a valid upper bound, even if a is not the minimizer.
    """
    if n < 0 or r < 0 or not 0 <= a <= n or not 0 < eta.lo <= eta.hi < Decimal("0.5"):
        raise ValueError("Invalid spectrum cutoff")
    arithmetic = eta.arithmetic
    two = arithmetic.number(2)
    first = (2 * (1 - eta)) ** n * (eta / (1 - eta)) ** a / (two ** r)
    probability = (1 - eta) ** n
    tail = arithmetic.number(0)
    for i in range(a):
        tail += probability
        if i + 1 < a:
            probability = probability * (n - i) / (i + 1) * eta / (1 - eta)
    return (first + tail).capped(1)


def support_sum_interval(r: int, q: int, T: int, kappa: int, arithmetic: IntervalArithmetic) -> Interval:
    """Direct finite sum Gamma in Eq. (75), with no asymptotic replacement.

    The ratio of binomial coefficients cancels factorials and is accumulated as
    a product. C(q,n) is updated as an exact integer. Huge population arrays are
    never constructed. All summands remain nonnegative.
    """
    if not 1 <= T <= min(q, r - 1) or kappa < 1 or q > (1 << (r - 1)):
        raise ValueError("Invalid finite-support parameters")
    if kappa == 1:
        return arithmetic.number(0)
    population = arithmetic.number(2) ** (r - 1)
    total = arithmetic.number(0)
    choose_q_n = 1
    for n in range(1, T + 1):
        choose_q_n = choose_q_n * (q - n + 1) // n
        ell = n // kappa + 1
        k = n - ell
        if k <= 0 or ((1 << (k - 1)) - k) < ell:
            continue
        closure = arithmetic.number(2) ** (k - 1)
        ratio = arithmetic.number(1)
        for i in range(ell):
            ratio = ratio * (closure - k - i) / (population - k - i)
        total += ratio * (choose_q_n * comb(n, ell))
        if total.lo >= 1:
            return arithmetic.number(1)
    return total.capped(1)


def evidence_interval(T: int, kappa: int, eta: Interval) -> Interval:
    if T < 1 or kappa < 1:
        raise ValueError("Positive count and support parameter required")
    return (1 + (1 - 2 * eta) ** kappa) ** (T // kappa) - 1


def _tau_scout(size: int, r: int, eta: float, test: bool) -> str:
    """Propose a threshold; the certificate re-evaluates its decimal value."""
    def criterion(tau):
        kl = tau * log(tau / eta) + (1 - tau) * log((1 - tau) / (1 - eta))
        capacity_nats = log(2) + tau * log(tau) + (1 - tau) * log(1 - tau)
        if test:
            terms = [-size * kl - log(2), log(size) - size * capacity_nats]
        else:
            # Distinctness is independent of tau and negligible on the paper grid.
            terms = [-size * kl, r * log(2) - size * capacity_nats]
        return float(logsumexp(terms))
    result = minimize_scalar(criterion, bounds=(eta + 1e-10, 0.5 - 1e-10),
                             method="bounded", options={"xatol": 1e-13})
    if not result.success:
        raise ArithmeticError("Scalar threshold search failed")
    return format(float(result.x), ".15g")


def selection_decoder_intervals(r: int, b: int, t: int, eta: Interval,
                                tau_test: str, tau_dec: str) -> tuple[Interval, Interval, Interval]:
    arithmetic = eta.arithmetic
    test, dec, two = arithmetic.number(tau_test), arithmetic.number(tau_dec), arithmetic.number(2)
    if not eta.hi < test.lo <= test.hi < Decimal("0.5") or not eta.hi < dec.lo <= dec.hi < Decimal("0.5"):
        raise ValueError("Disagreement thresholds must be strictly between eta and 1/2")
    n = b - t
    if n < 0:
        raise ValueError("Verification does not fit the total label budget")
    population = two ** (r - 1)
    distinct = 1 - arithmetic.number(n * t + comb(n, 2)) / population
    if distinct.lo <= 0:
        raise ValueError("The union-bound distinctness correction is not positive")
    ln2 = two.log()
    selection = ((-t * _kl(test, eta)).exp() / 2
                 + t * (-t * (ln2 - _natural_entropy(test))).exp()).capped(1)
    decoding = ((-n * _kl(dec, eta)).exp()
                + (two ** r - 2) / distinct * (-n * (ln2 - _natural_entropy(dec))).exp()).capped(1)
    return selection, decoding, distinct


def _integer_budget(r: int, eta: Interval) -> int:
    ln2 = eta.arithmetic.number(2).log()
    budget = r * (ln2 / (ln2 - _natural_entropy(eta)) + eta.arithmetic.number("0.5"))
    lower = int(budget.lo.to_integral_value(rounding=ROUND_FLOOR))
    upper = int(budget.hi.to_integral_value(rounding=ROUND_FLOOR))
    if lower != upper:
        raise ArithmeticError("Budget floor is not resolved; increase interval precision")
    return lower


def finite_certificate(r: int, method: str = "direct", precision: int = 80) -> dict:
    """Eq. (217)/(219), eta=.05, p=.5, q=r^2, b=floor((c+.5)r).

    ``strict_certificate`` is true only after comparison of an outward-rounded
    random lower endpoint and structured upper endpoint, BEFORE display rounding.
    """
    if method not in ("direct", "analytic"):
        raise ValueError("Certificate method must be direct or analytic")
    if method == "direct" and r not in DIRECT_CUTOFFS:
        raise ValueError("Direct preset exists only at r=768,1024,2048,4096 (Table 31)")
    arithmetic = IntervalArithmetic(precision)
    eta = arithmetic.number("0.05")
    q, t, b = r * r, subcube_size(r), _integer_budget(r, eta)
    if q > (1 << (r - 1)) or b - t > q - t:
        raise ValueError("The specified pool/budget is infeasible")
    if method == "direct":
        choices = DIRECT_CUTOFFS[r]
        T, kappa, a = choices["T"], choices["kappa"], choices["a"]
        tau_test, tau_dec = choices["tau_test"], choices["tau_dec"]
        gamma = support_sum_interval(r, q, T, kappa, arithmetic)
        failure = (2 * gamma).capped(1)
    else:
        T = 3 * r // 4
        # Verify the integer support cutoff with interval arithmetic too.
        k_interval = arithmetic.number(r - T - 1) / (2 * arithmetic.number(2 * q).log() / arithmetic.number(2).log())
        kappa = int(k_interval.lo.to_integral_value(rounding=ROUND_FLOOR))
        if kappa != int(k_interval.hi.to_integral_value(rounding=ROUND_FLOOR)) or kappa < 1:
            raise ValueError("Analytic support cutoff is unresolved or below one")
        if q > (1 << (r - 2)):
            raise ValueError("Analytic failure bound requires q <= m/2")
        gamma = arithmetic.number(1) / (2 * q - 1)
        failure = 2 * gamma
        _, a = information_spectrum(b - T, r, 0.05)
        tau_test = _tau_scout(t, r, 0.05, True)
        tau_dec = _tau_scout(b - t, r, 0.05, False)
    Trec = b - T
    if not 1 <= T <= Trec or T >= r:
        raise ValueError("The allocation cutoffs are infeasible")
    evidence = evidence_interval(T, kappa, eta)
    psi = spectrum_interval(Trec, r, eta, a)
    selection, decoding, distinct = selection_decoder_intervals(r, b, t, eta, tau_test, tau_dec)
    gain = (1 - 2 * eta) ** 2 / 8
    energy = (arithmetic.number("0.5") + evidence / 2 + failure + psi).capped(1)
    lower = arithmetic.number("0.25") - arithmetic.number(b) / (4 * arithmetic.number(2) ** r) - gain * energy
    upper = (arithmetic.number("0.25") - gain + selection + decoding).capped("0.25")
    lower_endpoint = max(Decimal(0), lower.lo)
    upper_endpoint = upper.hi
    quantum = Decimal("0.000001")
    return dict(r=r, q=q, eta="0.05", prior="0.5", b=b, t=t, n=b-t, T=T, Trec=Trec,
                kappa=kappa, a=a, tau_test=tau_test, tau_dec=tau_dec, method=method,
                precision=precision, gamma_upper=str(gamma.hi), failure_upper=str(failure.hi),
                evidence_upper=str(evidence.hi), psi_upper=str(psi.hi),
                selection_upper=str(selection.hi), decoding_upper=str(decoding.hi),
                distinct_lower=str(distinct.lo), random_lower=str(lower_endpoint),
                structured_upper=str(upper_endpoint),
                random_lower_display=str(lower_endpoint.quantize(quantum, rounding=ROUND_FLOOR)),
                structured_upper_display=str(upper_endpoint.quantize(quantum, rounding=ROUND_CEILING)),
                strict_certificate=lower_endpoint > upper_endpoint,
                interpretation="pool-law expected-risk bound; not a simulated population risk")


def expected_affine_flats(d: int, q: int | None = None) -> Fraction:
    """Exact rational Eq. (214) for contained eight-point affine flats."""
    m = 1 << d
    q = (d + 1) ** 2 if q is None else int(q)
    if d < 3 or not 0 <= q <= m:
        raise ValueError("Require d >=3 and 0 <= q <= 2^d")
    if q < 8:
        return Fraction(0)
    total_flats = (1 << (d - 3)) * math.prod(m - (1 << i) for i in range(3)) // (7 * 6 * 4)
    return Fraction(total_flats * math.prod(q - i for i in range(8)), math.prod(m - i for i in range(8)))


def short_dependency_bound(r: int, precision: int = 80) -> dict:
    """Eq. (221): exact integer numerator with a directed-rounded quotient."""
    q, t = r * r, subcube_size(r)
    h = max(t, floor(r / (4 * log2(q))))
    m = 1 << (r - 1)
    if q >= m:
        raise ValueError("Short-dependency bound requires q < m")
    total, choose = 0, 1
    for w in range(1, h + 1):
        choose = choose * (q - w + 1) // w
        if w >= 4 and w % 2 == 0:
            total += choose
    arithmetic = IntervalArithmetic(precision)
    rho = (arithmetic.number(2 * total) / (m - q)).capped(1)
    error_lower = arithmetic.number("0.5") - rho / 2
    return dict(r=r, q=q, t=t, h=h, rho_upper=str(rho.hi),
                random_testing_error_lower=str(error_lower.lo), random_testing_error_upper="0.5")
