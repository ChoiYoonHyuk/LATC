"""Small directed-rounding interval arithmetic for finite risk certificates.

Inputs are exact integers or decimal strings (never binary floats). Basic
operations round toward -infinity/+infinity. Decimal.ln/exp are correctly rounded
to nearest; taking an adjacent representable number on each side encloses the
exact result. No conclusion is based on rounding a float to six display digits.

This is a deliberately small internal implementation, not a general-purpose
interval library. It uses only the operations needed in Appendix P.4.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN


class IntervalArithmetic:
    def __init__(self, precision: int = 80):
        if precision < 30:
            raise ValueError("Certificate precision must be at least 30 decimal digits")
        self.precision = precision
        self.down = Context(prec=precision, rounding=ROUND_FLOOR, Emax=999999999, Emin=-999999999)
        self.up = Context(prec=precision, rounding=ROUND_CEILING, Emax=999999999, Emin=-999999999)
        self.near = Context(prec=precision, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999)

    def number(self, value: int | str | Decimal | Interval) -> Interval:
        if isinstance(value, Interval):
            if value.arithmetic is not self:
                raise ValueError("Cannot mix interval contexts")
            return value
        if not isinstance(value, (int, str, Decimal)):
            raise TypeError("Use exact integers or decimal strings, not float inputs")
        d = Decimal(value)
        if not d.is_finite():
            raise ValueError("Finite interval inputs required")
        return Interval(d, d, self)


@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal
    arithmetic: IntervalArithmetic

    def __post_init__(self):
        if self.lo > self.hi:
            raise ArithmeticError("Reversed interval endpoints")

    def _coerce(self, other):
        return self.arithmetic.number(other)

    def __add__(self, other):
        other, a = self._coerce(other), self.arithmetic
        return Interval(a.down.add(self.lo, other.lo), a.up.add(self.hi, other.hi), a)

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate(), self.arithmetic)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other, a = self._coerce(other), self.arithmetic
        if self.lo >= 0 and other.lo >= 0:
            return Interval(a.down.multiply(self.lo, other.lo), a.up.multiply(self.hi, other.hi), a)
        pairs = [(x, y) for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(min(a.down.multiply(x, y) for x, y in pairs),
                        max(a.up.multiply(x, y) for x, y in pairs), a)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other, a = self._coerce(other), self.arithmetic
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Interval divisor contains zero")
        if self.lo >= 0 and other.lo > 0:
            return Interval(a.down.divide(self.lo, other.hi), a.up.divide(self.hi, other.lo), a)
        inverse = Interval(a.down.divide(Decimal(1), other.hi), a.up.divide(Decimal(1), other.lo), a)
        return self * inverse

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, exponent: int):
        if not isinstance(exponent, int):
            raise TypeError("Only integer powers are supported; use exp/log for other powers")
        if exponent < 0:
            return 1 / (self ** (-exponent))
        result, base, n = self._coerce(1), self, exponent
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def log(self):
        a = self.arithmetic
        if self.lo <= 0:
            raise ValueError("Logarithm requires a positive interval")
        if self.lo == self.hi == 1:
            return self._coerce(0)
        lo, hi = a.near.ln(self.lo), a.near.ln(self.hi)
        return Interval(a.down.next_minus(lo), a.up.next_plus(hi), a)

    def exp(self):
        a = self.arithmetic
        if self.lo == self.hi == 0:
            return self._coerce(1)
        lo, hi = a.near.exp(self.lo), a.near.exp(self.hi)
        return Interval(a.down.next_minus(lo), a.up.next_plus(hi), a)

    def capped(self, cap: int | str = 1):
        cap = self._coerce(cap)
        return Interval(min(self.lo, cap.lo), min(self.hi, cap.hi), self.arithmetic)

    def endpoints(self) -> dict[str, str]:
        return {"lower": str(self.lo), "upper": str(self.hi)}
