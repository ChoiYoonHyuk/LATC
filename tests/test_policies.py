import numpy as np
import pytest
from latc.affine import sample_target_array
from latc.paired import (paired_pools, AnnotationOracle, acquire, evaluate,
                         POLICIES, summarize_records)

@pytest.mark.parametrize('d',[8,12])
def test_equal_cardinality_coupled_pools(d):
    pools,common=paired_pools(d,10**12+1234)
    for j in (0,1):
        for name in ('random','structured'):
            p=pools[name][j]
            assert len(p.coordinates)==(d+1)**2
            assert len(np.unique(p.coordinates))==(d+1)**2
            assert not np.any(common[j,p.coordinates])
        np.testing.assert_array_equal(pools['structured'][j].probe,np.arange(8))
    assert common.any()

@pytest.mark.parametrize('d',[8,12])
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('pool_name',['random','structured'])
def test_exact_budget_eligibility_and_evaluation_separation(d,policy,pool_name):
    target=sample_target_array(d,.05,123456+d)
    pools,common=paired_pools(d,10**12+123456+d)
    b=int(1.7*(d+1))
    row,history=evaluate(target,pools[pool_name],common,policy,b)
    assert len(history)==b
    assert len({(q['region'],q['coordinate']) for q in history})==b
    assert row['batches']<=2
    assert row['unqueried']==pytest.approx(row['full']*target.labels.size/(target.labels.size-b))
    if policy=='prescribed_latc':
        assert sum(q['batch']==1 for q in history)==8
        second=[q for q in history if q['batch']==2]
        assert all(q['region']==row['selected_region'] for q in second)


def test_oracle_prevalidates_complete_batch_without_charging_on_failure():
    target=sample_target_array(8,.1,12);pools,_=paired_pools(8,123)
    oracle=AnnotationOracle(target.labels,pools['random'],5)
    x=int(pools['random'][0].coordinates[0])
    with pytest.raises(ValueError):oracle.query_batch([(0,x),(0,x)])
    assert len(oracle.history)==0
    oracle.query_batch([(0,x)])
    with pytest.raises(ValueError):oracle.query_batch([(0,x)])
    assert len(oracle.history)==1


def test_budgets_share_the_first_stage_localization():
    target=sample_target_array(12,.025,12531);pools,_=paired_pools(12,12531+10**12)
    answers=[]
    for b in (19,22,24):
        oracle=AnnotationOracle(target.labels,pools['structured'],b)
        _,selected,lr=acquire('prescribed_latc',12,.025,pools['structured'],oracle,b)
        answers.append((selected,lr,oracle.history[:8]))
    assert answers[0]==answers[1]==answers[2]


def test_paired_interval_uses_differences_not_independent_standard_errors():
    rows=[]
    for i in range(8):
        common={'protocol':'window','d':8,'eta':.05,'budget':17,'policy':'concentrated',
                'replicate':i,'target_seed':i,'common_hash':str(i)}
        for pool,offset in [('random',.01),('structured',0)]:
            loss=.1+.01*i+offset
            rows.append({**common,'pool':pool,'common':loss,'full':loss,'unqueried':loss})
    _,contrasts,_=summarize_records(rows)
    for contrast in contrasts:
        assert contrast['difference']==pytest.approx(.01)
        assert contrast['ci95_halfwidth']<1e-15
