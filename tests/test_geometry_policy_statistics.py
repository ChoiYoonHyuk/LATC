from itertools import combinations
import numpy as np
import pytest
from latc.algebra import gf2_rank,likelihood_table
from latc.pools import find_affine_flat,is_affine_flat,paired_pool_designs,two_disjoint_checks
from latc.policies import generate_population,indexed_rng,prescribed_two_batch,fixed_control
from latc.statistics import equal_prior_error_interval,holm_adjust,student_summary


def brute_flat_exists(pool):
    available=set(map(int,pool))
    for anchor in available:
        for u,v,w in combinations([x^anchor for x in available if x!=anchor],3):
            if gf2_rank([u,v,w])==3 and {anchor^x for x in (0,u,v,w,u^v,u^w,v^w,u^v^w)}<=available:
                return True
    return False


@pytest.mark.parametrize("seed",range(12))
def test_exhaustive_flat_search_matches_independent_enumeration(seed):
    pool=np.random.default_rng(seed).choice(16,size=8+seed%5,replace=False)
    result=find_affine_flat(pool)
    assert (result is not None)==brute_flat_exists(pool)
    if result is not None:
        assert is_affine_flat(result)
        np.testing.assert_allclose(np.sort(likelihood_table(result,.1)),np.sort(likelihood_table(range(8),.1)),atol=1e-12)


@pytest.mark.parametrize("d",[8,12])
def test_pool_coupling_exact_budget_persistence_and_shared_targets(d):
    designs,targets=paired_pool_designs(d,np.random.default_rng(d),include_flat=True)
    labels,active,theta=generate_population(d,.05,np.random.default_rng(d+2))
    q=(d+1)**2
    for design in designs.values():
        assert all(len(p)==q for p in design.pools)
        assert all(np.intersect1d(t,p).size==0 for t,p in zip(targets,design.pools))
        for b in [int(1.5*(d+1)),int(1.9*(d+1))]:
            run=prescribed_two_batch(d,.05,.5,design,labels,b)
            assert sum(map(len,run.queries))==b
            assert all(len(np.unique(p))==len(p) for p in run.queries)
            for j in (0,1):
                np.testing.assert_array_equal(run.labels[j],labels[j,run.queries[j]])
                assert np.intersect1d(run.queries[j],targets[j]).size==0
            if run.selected_region==0:
                assert len(run.queries[1])==0
            else:
                np.testing.assert_array_equal(run.queries[0],design.blocks[0])
                assert np.intersect1d(run.queries[1],design.blocks[1]).size==0
            components=run.prediction.decomposition(targets)
            assert components['conditional_brier']==pytest.approx(components['residual']+components['parameter_uncertainty']+components['indicator_uncertainty'])
    fixed=fixed_control(d,.05,.5,designs['prefix'],labels,13)
    np.testing.assert_array_equal(fixed.queries[0],designs['prefix'].remainders[0][:13])


def test_rng_key_is_replicate_and_component_specific():
    a=indexed_rng(2026,512,8,.05,1,1).integers(2**30,size=20)
    b=indexed_rng(2026,512,8,.05,1,1).integers(2**30,size=20)
    c=indexed_rng(2026,512,8,.05,1,2).integers(2**30,size=20)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    with pytest.raises(ValueError):
        indexed_rng(2026,512,8,.01234567,1,1)


def test_statistical_conventions():
    result=student_summary([1,2,3,4])
    assert result['mean']==2.5
    assert result['degrees_of_freedom']==3
    assert student_summary([1,2,3,4],family_size=6)['half_width']>result['half_width']
    np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])
    interval=equal_prior_error_interval(0,0,131072)
    assert interval['estimated_error']==interval['ci_lower']==0
    assert interval['ci_upper']==pytest.approx(3.344e-5,abs=1e-8)
    assert equal_prior_error_interval(3,1,131072)['estimated_error']==4/262144


def test_invalid_pool_cardinality_is_not_silently_clipped():
    with pytest.raises(ValueError):
        paired_pool_designs(3,np.random.default_rng(1))
