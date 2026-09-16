from fractions import Fraction
import csv
from pathlib import Path
import numpy as np
import pytest
from latc.noiseless import NoiselessSolver,basis_optimum
from latc.exhaustive import solve_exhaustive,POOLS
from latc.mechanisms import (relation_statistics,RELATION_PROBES,binomial_diagnostic,
                             terminal_components,terminal_gap,dependency_code)
from latc.rational import rational_optima

@pytest.mark.parametrize('d,b,values,states',[
    (8,17,[.20432,.18986,.13139,.13081,.20366],1588),
    (12,21,[.22953,.18790,.12935,.12870,.21785],4171)])
def test_rq1_operating_points(d,b,values,states):
    solver=NoiselessSolver(d);r=solver.solve_budget(b)
    for method,expected in zip(('uniform','fixed','latc','adaptive','basis'),values):
        assert r[method+'_unqueried']==pytest.approx(expected,abs=5e-6)
    assert r['adaptive_cache_states']==states
    assert r['probe_dimension']==3 and r['probe_labels']==8
    solver.clear_caches()

@pytest.mark.parametrize('eta',[0,.05,.1,.2])
def test_basis_pool_all_classes_agree_with_analytic_law(eta):
    result=solve_exhaustive(2,(0,1,2),eta)
    for b in range(7):
        expected,_=basis_optimum(2,b,eta)
        assert result.fixed[b]==pytest.approx(expected,abs=1e-12)
        assert result.two_batch[b]==pytest.approx(expected,abs=1e-12)
        assert result.adaptive[b]==pytest.approx(expected,abs=1e-12)


def test_reduced_state_matches_independent_full_history_solver():
    result=solve_exhaustive(2,(0,1,2,3),0)
    expected=[Fraction(1,4)]*3+[Fraction(19,80),Fraction(15,64),Fraction(11,48),Fraction(3,16),Fraction(1,6)]
    for b in range(8):
        solver=NoiselessSolver(2);r=solver.solve_budget(b)
        assert result.fixed[b]*8/(8-b)==pytest.approx(float(expected[b]),abs=1e-12)
        assert result.fixed[b]==pytest.approx(r['fixed_full'],abs=1e-12)
        assert result.adaptive[b]==pytest.approx(r['adaptive_full'],abs=1e-12)
        solver.clear_caches()

@pytest.mark.slow
@pytest.mark.parametrize('eta',[0,.05,.1,.2])
@pytest.mark.parametrize('pool_name',['short','long'])
def test_all_rq3_printed_values_and_full_class_constraints(eta,pool_name):
    result=solve_exhaustive(4,POOLS[pool_name],eta)
    reference=Path(__file__).resolve().parents[1]/'reference'/'rq3_manuscript.csv'
    with reference.open() as f:refs=list(csv.DictReader(f))
    for ref in refs:
        if ref['pool']!=pool_name or float(ref['eta'])!=eta:continue
        b=int(ref['budget'])
        for method in ('fixed','two_batch','adaptive'):
            assert getattr(result,method)[b]*32/(32-b)==pytest.approx(float(ref[method]),abs=5e-7)
    assert result.joint_states==531441 and result.marginal_pairs==16777216
    assert np.all(result.adaptive<=result.two_batch+1e-12)
    assert np.all(result.two_batch<=result.fixed+1e-12)
    assert np.all(np.diff(result.adaptive)<=1e-12)
    assert np.all(np.diff(result.fixed)<=1e-12)

    np.testing.assert_allclose(result.two_batch,result.adaptive,atol=1e-12)
    if pool_name=='long':np.testing.assert_allclose(result.fixed,result.adaptive,atol=1e-12)
    k=4 if pool_name=='short' else 6
    assert result.fixed[12]*32/20==pytest.approx(terminal_components(eta,k)['risk_unqueried'],abs=1e-12)


def test_rational_complete_target_law_matches_noisy_solver():
    exact=rational_optima(2,(0,1,2),Fraction(1,10))
    numeric=solve_exhaustive(2,(0,1,2),.1)
    for method in ('fixed','two_batch','adaptive'):
        np.testing.assert_allclose(getattr(numeric,method),[float(x) for x in exact[method]],atol=1e-12)

@pytest.mark.parametrize('eta',[0,.05,.1,.2,.49])
def test_relation_length_laws(eta):
    for name,k in [('length_four',4),('length_eight',8)]:
        s=relation_statistics(RELATION_PROBES[name],eta)
        assert s['count']==8 and s['rank']==7
        assert s['dependency_weights']==[k]
        assert s['testing_error']==pytest.approx(.5-(1-2*eta)**k/4,abs=1e-13)
    assert relation_statistics(RELATION_PROBES['independent'],eta)['testing_error']==.5


def test_subcube_complete_dependency_spectrum():
    weights=[v.bit_count() for v in dependency_code(tuple(range(8))) if v]
    assert weights.count(4)==14 and weights.count(8)==1

@pytest.mark.parametrize('L',[6,16,64,256])
def test_noise_diagnostic_stability_at_tiny_absolute_eta(L):
    zero=binomial_diagnostic(L,0)
    expected=2**(-zero['checks']-1)
    assert zero['testing_error']==pytest.approx(expected,abs=1e-14)
    small=binomial_diagnostic(L,.001)
    larger=binomial_diagnostic(L,1)
    assert 0<=small['testing_error']<=larger['testing_error']<=.5+1e-14
    assert small['parity_signal']>larger['parity_signal']

@pytest.mark.parametrize('eta',[0,.01,.05,.1,.2,.25,.49])
def test_terminal_ordering_and_decomposition(eta):
    a,b=terminal_components(eta,4),terminal_components(eta,6)
    assert a['risk_unqueried']-b['risk_unqueried']==pytest.approx(terminal_gap(eta),abs=1e-12)
    if eta>0:
        assert a['task_uncertainty']>b['task_uncertainty']
        assert a['reliability_uncertainty']<b['reliability_uncertainty']
        assert a['risk_unqueried']>b['risk_unqueried']
