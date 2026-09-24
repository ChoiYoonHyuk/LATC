"""Independent complete target-array enumeration with rational model probabilities.

This validator never calls the production posterior or reduced-state transition
formulas to compute its terminal costs or acquisition values.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import numpy as np
import pytest
from latc.exact_noiseless import NoiselessOptimizer,basis_risk
from latc.exact_noisy import ExactNoisyOptimizer,SHORT_POOL,LONG_POOL,COMMON_TARGETS
from latc.comparisons import matched_common_brier,common_brier_gap,complete_target_scores


def reference_optimizer(d,eta=Fraction(0)):
    m=1<<d
    regional=[]
    for labels in range(1<<m):
        probability=Fraction(0)
        for beta in range(2):
            for u in range(m):
                e=sum(((labels>>x)&1)!=(beta^((u&x).bit_count()&1)) for x in range(m))
                probability+=(1-eta)**(m-e)*eta**e/(2*m)
        regional.append(probability)
    codes=np.arange(1<<(2*m),dtype=np.int64)
    binary=((codes[:,None]>>np.arange(2*m))&1).astype(float)
    mass=np.array([float((regional[c&((1<<m)-1)]+regional[c>>m])/(2*(1<<m))) for c in codes])
    assert sum(regional)==1 and mass.sum()==pytest.approx(1)
    @lru_cache(None)
    def cost(mask,y):
        compatible=(codes&mask)==y
        probability=mass[compatible].sum()
        if probability==0:
            return 0.0
        ones=mass[compatible]@binary[compatible]
        return float(np.sum(ones*(probability-ones)/probability)/(2*m))
    def outcomes(subset):
        for values in product((0,1),repeat=len(subset)):
            yield sum(v<<i for i,v in zip(subset,values))
    @lru_cache(None)
    def fixed(mask,y,remaining):
        available=[i for i in range(2*m) if not mask>>i&1]
        return min(sum(cost(mask|sum(1<<i for i in subset),y|assignment) for assignment in outcomes(subset))
                   for subset in combinations(available,remaining))
    @lru_cache(None)
    def adaptive(mask,y,remaining):
        if remaining==0:
            return cost(mask,y)
        return min(adaptive(mask|1<<i,y,remaining-1)+adaptive(mask|1<<i,y|1<<i,remaining-1)
                   for i in range(2*m) if not mask>>i&1)
    def two_batch(budget):
        return min(sum(fixed(sum(1<<i for i in subset),assignment,budget-n) for assignment in outcomes(subset))
                   for n in range(budget+1) for subset in combinations(range(2*m),n))
    return fixed,adaptive,two_batch


@pytest.mark.parametrize("d,b",[(1,1),(1,3),(2,3),(2,5),(2,6)])
def test_complete_arrays_match_reduced_noiseless_optima(d,b):
    fixed,adaptive,two_batch=reference_optimizer(d)
    solver=NoiselessOptimizer(d)
    assert solver.fixed(b,False)==pytest.approx(fixed(0,0,b),abs=2e-13)
    assert solver.adaptive(b,False)==pytest.approx(adaptive(0,0,b),abs=2e-13)
    assert solver.two_batch(b,False)==pytest.approx(two_batch(b),abs=2e-13)
    assert solver.uniform(b,False)==pytest.approx(solver.uniform_recursive(b,False),abs=2e-13)


def test_seven_point_first_batch_exact_fraction():
    solver=NoiselessOptimizer(8)
    assert solver.two_batch(16)==pytest.approx(float(Fraction(184981,1364992)),abs=2e-13)
    profile=solver.two_batch_profile(16)
    best=min(row['population_risk'] for row in profile)
    assert [row['first_batch'] for row in profile if abs(row['population_risk']-best)<1e-13]==[7]
    assert solver.subcube(16)[0]>solver.two_batch(16)


@pytest.mark.parametrize("eta",[0,.05,.1,.2])
@pytest.mark.parametrize("prior",[0,.1,.5,.9,1])
def test_matched_complete_acquisition_closed_form(eta,prior):
    # Direct integration via the production exact likelihood table, independent of
    # the simplified closed-form function being tested.
    values=[]
    for length,pool in [(4,SHORT_POOL),(6,LONG_POOL)]:
        solver=ExactNoisyOptimizer(4,(pool,pool),eta,prior,(COMMON_TARGETS,COMMON_TARGETS))
        codes=solver.complete_codes(tuple(range(12)))
        risk=float(solver.costs['common'][codes].sum())
        assert risk==pytest.approx(matched_common_brier(length,eta,prior),abs=2e-12)
        values.append(risk)
        scores=complete_target_scores(solver)
        assert scores['classification']==pytest.approx(.5-(1-2*eta)**4/4,abs=2e-12)
        assert scores['confidence_moment_1']==pytest.approx((1-2*eta)**4/2,abs=2e-12)
    if eta>0:
        assert values[0]>values[1]
    if prior==.5:
        assert values[0]-values[1]==pytest.approx(common_brier_gap(eta),abs=2e-12)


def test_basis_formula_and_independent_first_batch_tie_effect():
    stay=basis_risk(8,5,.1,.5,False)
    # A split into t=2 and n=3 has only one length-three unqueried representation.
    switched=.25-(5+.25*.8**8)/(8*256)
    assert switched>stay
    solver=NoiselessOptimizer(8)
    assert solver.fixed(9)>=basis_risk(8,9,0,.5)-1e-13  # rank-9 full basis is optimal here


def test_objectives_use_same_posterior_and_terminal_population():
    solver=ExactNoisyOptimizer(4,(SHORT_POOL,SHORT_POOL),.05,.5,(COMMON_TARGETS,COMMON_TARGETS))
    bald=solver.sequential_objective('bald')
    epig=solver.sequential_objective('epig')
    brier=solver.sequential_objective('brier')
    np.testing.assert_allclose(epig,brier,atol=2e-12)
    assert epig[6]<bald[6]
    assert bald[10]<epig[10]
    for b in [0,6,9,12]:
        optimum=solver.optimize(b)
        assert bald[b]>=optimum['adaptive']-2e-12
        assert epig[b]>=optimum['adaptive']-2e-12
        assert solver.batchbald(b)>=optimum['two_batch']-2e-12
    assert bald[-1]==pytest.approx(epig[-1],abs=2e-12)


def test_noisy_enumeration_input_and_memory_guards():
    with pytest.raises(ValueError,match="integral"):
        ExactNoisyOptimizer(4,([0,1.5],[0,1]),.1)
    with pytest.raises(ValueError,match="memory guard"):
        ExactNoisyOptimizer(20,(range(6),range(6)),.1)
    with pytest.raises(ValueError,match="guard exceeded"):
        ExactNoisyOptimizer(4,(range(7),range(6)),.1)
