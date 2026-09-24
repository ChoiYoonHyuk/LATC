"""Binary linear algebra and unnormalized Walsh-Hadamard transforms.

Coordinates are integers with little-endian binary digits.  An augmented row is
``1 | (x << 1)``; bit zero is the affine intercept throughout the package.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Iterable
import numpy as np


def gf2_rank(rows: Iterable[int]) -> int:
    pivots: dict[int, int] = {}
    for row in rows:
        v = int(row)
        if v < 0:
            raise ValueError("GF(2) rows must be nonnegative integer bit masks")
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    return len(pivots)


def augmented_rank(coordinates: Iterable[int]) -> int:
    return gf2_rank(1 | (int(x) << 1) for x in coordinates)


def dependency_basis(coordinates: Iterable[int]) -> tuple[int, ...]:
    """Return an independent basis of ker(A.T), encoded as query-index masks."""
    pivots: dict[int, tuple[int, int]] = {}
    dependencies = []
    for i, x in enumerate(coordinates):
        if int(x) < 0:
            raise ValueError("Coordinates must be nonnegative")
        row, witness = 1 | (int(x) << 1), 1 << i
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                a, b = pivots[p]
                row ^= a
                witness ^= b
            else:
                pivots[p] = (row, witness)
                break
        if row == 0:
            dependencies.append(witness)
    return tuple(dependencies)


def span_elements(basis: Iterable[int], max_dimension: int = 22) -> list[int]:
    basis = tuple(map(int, basis))
    if len(basis) > max_dimension:
        raise ValueError("Explicit span enumeration would exceed the dimension guard")
    out = [0]
    for v in basis:
        out += [u ^ v for u in out]
    return out


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized transform along the last axis; never modifies the input."""
    a = np.array(values, copy=True)
    if a.ndim == 0 or a.shape[-1] == 0 or a.shape[-1] & (a.shape[-1] - 1):
        raise ValueError("The last dimension must be a positive power of two")
    # Unsigned/bool inputs cannot represent negative correlations.
    if a.dtype.kind in "bu":
        a = a.astype(np.int64)
    n, h = a.shape[-1], 1
    while h < n:
        view = a.reshape(*a.shape[:-1], -1, 2 * h)
        left = view[..., :h].copy()
        right = view[..., h:].copy()
        view[..., :h] = left + right
        view[..., h:] = left - right
        h *= 2
    return a


def affine_values(d: int, parameter: int) -> np.ndarray:
    """Clean labels of theta=(beta,u), encoded as beta + 2*u."""
    if d < 0 or d > 22 or not 0 <= parameter < (1 << (d + 1)):
        raise ValueError("Invalid dimension/parameter (explicit-population guard: d <= 22)")
    beta, slope = parameter & 1, parameter >> 1
    return np.fromiter((beta ^ ((x & slope).bit_count() & 1)
                        for x in range(1 << d)), dtype=np.uint8, count=1 << d)


@lru_cache(maxsize=2048)
def _likelihood_table(n: int, basis: tuple[int, ...], eta: float) -> np.ndarray:
    if n > 20:
        raise ValueError("Outcome-table enumeration is restricted to <=20 labels")
    if not 0 <= eta < 0.5:
        raise ValueError("Require 0 <= eta < 1/2")
    coefficients = np.zeros(1 << n)
    delta = 1 - 2 * eta
    for v in span_elements(basis):
        coefficients[v] = delta ** v.bit_count()
    result = fwht(coefficients)
    # Fourier cancellation can create tiny negative values at eta=0.
    if result.min(initial=0) < -1e-10:
        raise ArithmeticError("Negative affine likelihood")
    result = np.maximum(result, 0)
    result.setflags(write=False)
    return result


def likelihood_table(coordinates: Iterable[int], eta: float) -> np.ndarray:
    """Eq. (3) for all outcomes; outcome bit i labels query i."""
    q = tuple(map(int, coordinates))
    if len(set(q)) != len(q):
        raise ValueError("Repeated query identities are forbidden")
    return _likelihood_table(len(q), dependency_basis(q), float(eta))


def outcome_code(labels: Iterable[int]) -> int:
    result = 0
    for i, y in enumerate(labels):
        if y not in (0, 1):
            raise ValueError("Labels must be binary")
        result |= int(y) << i
    return result


def syndrome_likelihood(coordinates: Iterable[int], labels: Iterable[int], eta: float) -> float:
    """Appendix I.2: nonnegative dynamic programming on dependency syndromes."""
    q, y = tuple(coordinates), tuple(labels)
    if len(q) != len(y) or not 0 <= eta < 0.5:
        raise ValueError("Invalid observations or noise")
    basis = dependency_basis(q)
    if len(basis) > 22:
        raise ValueError("Syndrome state space too large")
    code = outcome_code(y)
    target = sum(((v & code).bit_count() & 1) << j for j, v in enumerate(basis))
    size = 1 << len(basis)
    prob = np.zeros(size)
    prob[0] = 1
    index = np.arange(size)
    for i in range(len(q)):
        column = sum(((v >> i) & 1) << j for j, v in enumerate(basis))
        prob = (1 - eta) * prob + eta * prob[index ^ column]
    return float(size * prob[target])
