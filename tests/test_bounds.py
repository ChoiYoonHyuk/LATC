from decimal import Decimal
from fractions import Fraction
from math import comb
import pytest
from latc.bounds import (finite_certificate,information_spectrum,short_dependency_bound,
                         spectrum_interval,support_sum_interval,expected_affine_flats,subcube_size)
from latc.intervals import IntervalArithmetic


def test_interval_arithmetic_encloses_rational_operations():
    arithmetic=IntervalArithmetic(40)
    a=arithmetic.number(1)/7
    b=arithmetic.number(2)/13
    for interval,exact in [(a+b,Fraction(1,7)+Fraction(2,13)),(a-b,Fraction(1,7)-Fraction(2,13)),
                           (a*b,Fraction(1,7)*Fraction(2,13)),(a/b,Fraction(13,14)),
                           (a**7,Fraction(1,7)**7)]:
        assert Fraction(interval.lo)<=exact<=Fraction(interval.hi)
    two=arithmetic.number(2)
    result=two.log().exp()
    assert result.lo<=2<=result.hi
    with pytest.raises(TypeError):
        arithmetic.number(.05)


def test_finite_support_sum_against_exact_fractions():
    r,q,T,kappa=9,25,7,4
    total=Fraction(0)
    for n in range(1,T+1):
        ell=n//kappa+1
        k=n-ell
        if k>=1 and (1<<(k-1))-k>=ell:
            total+=Fraction(comb(q,n)*comb(n,k)*comb((1<<(k-1))-k,ell),comb((1<<(r-1))-k,ell))
    total=min(Fraction(1),total)
    interval=support_sum_interval(r,q,T,kappa,IntervalArithmetic(50))
    assert Fraction(interval.lo)<=total<=Fraction(interval.hi)


def test_strict_information_spectrum_event_and_zero_noise():
    n,r,a=8,9,2
    arithmetic=IntervalArithmetic(50)
    result=spectrum_interval(n,r,arithmetic.number('0.05'),a)
    exact=Fraction(19**(n-a),10**n*2**r)+sum(Fraction(comb(n,i)*19**(n-i),20**n) for i in range(a))
    exact=min(Fraction(1),exact)
    assert Fraction(result.lo)<=exact<=Fraction(result.hi)
    numeric,cutoff=information_spectrum(n,r,.05)
    assert 0<=cutoff<=n and 0<=numeric<=1
    assert information_spectrum(3,9,0)[0]==2**-6
    assert information_spectrum(0,9,.05)[0]==2**-9


@pytest.mark.parametrize("r,lower,upper,strict",[(768,'0.148749','0.149029',False),
    (1024,'0.173369','0.149029',True),(2048,'0.198141','0.149029',True),
    (4096,'0.199370','0.148751',True)])
def test_table33_direct_certificates(r,lower,upper,strict):
    result=finite_certificate(r)
    assert result['random_lower_display']==lower
    assert result['structured_upper_display']==upper
    assert result['strict_certificate']==strict
    assert strict==(Decimal(result['random_lower'])>Decimal(result['structured_upper']))


@pytest.mark.parametrize("r,expected",[(1024,'4e-12'),(4096,'1e-523'),(262144,'1e-64334')])
def test_table34_whole_pool_short_dependency_bound(r,expected):
    result=short_dependency_bound(r)
    assert Decimal(result['rho_upper'])<=Decimal(expected)
    assert result['t']==subcube_size(r)
    assert result['h']>=result['t']


def test_affine_flat_expectation():
    assert expected_affine_flats(3,8)==1
    assert float(expected_affine_flats(8))==pytest.approx(244.0962644967537)
    assert float(expected_affine_flats(16))<.00179


@pytest.mark.slow
@pytest.mark.parametrize("r,lower,upper",[(256,'0.148749','0.199426'),(1024,'0.148749','0.149029'),
    (4096,'0.148749','0.148751'),(16384,'0.193510','0.148751'),
    (65536,'0.199374','0.148751'),(262144,'0.199374','0.148751')])
def test_table29_analytic_certificates(r,lower,upper):
    result=finite_certificate(r,'analytic')
    assert result['random_lower_display']==lower
    assert result['structured_upper_display']==upper
