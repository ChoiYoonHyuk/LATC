import csv
from pathlib import Path
import pytest
from latc.exact_noiseless import NoiselessOptimizer
from latc.exact_noisy import ExactNoisyOptimizer,SHORT_POOL,LONG_POOL,COMMON_TARGETS

REFERENCE=Path(__file__).resolve().parents[1]/'reference'


@pytest.mark.slow
@pytest.mark.parametrize('d',[8,12])
def test_complete_noiseless_paper_tables(d):
    with (REFERENCE/'noiseless_tables_5_6.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if int(r['d'])==d]
    solver=NoiselessOptimizer(d)
    for row in rows:
        b=int(row['b'])
        values=dict(uniform=solver.uniform(b),fixed=solver.fixed(b),subcube=solver.subcube(b)[0],
                    two_batch=solver.two_batch(b),adaptive=solver.adaptive(b))
        for key,value in values.items():
            assert value==pytest.approx(float(row[key]),abs=5.01e-7), (d,b,key,value,row[key])


@pytest.mark.slow
@pytest.mark.parametrize('eta',[0,.05,.10,.20])
@pytest.mark.parametrize('length',[4,6])
def test_all_noisy_paper_tables(eta,length):
    with (REFERENCE/'noisy_tables_18_21.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if float(r['eta'])==eta]
    pool=SHORT_POOL if length==4 else LONG_POOL
    solver=ExactNoisyOptimizer(4,(pool,pool),eta,.5,(COMMON_TARGETS,COMMON_TARGETS))
    for row in rows:
        b=int(row['b'])
        for metric in ['common','population']:
            values=solver.optimize(b,metric)
            prefix='common' if metric=='common' else 'unqueried'
            for method,value in values.items():
                suffix='long_all' if length==6 else 'short_fixed' if method=='fixed' else 'short_two_ad'
                if metric=='population':
                    value *= 32/(32-b)
                assert value==pytest.approx(float(row[f'{prefix}_{suffix}']),abs=5.01e-7), (eta,length,b,metric,method,value)
