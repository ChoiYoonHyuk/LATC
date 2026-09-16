from __future__ import annotations
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import csv
import json
import math
import numpy as np
from .affine import (regional_posterior, dense_regional_posterior, parity_table,
                     gf2_rank, sample_target_array)
from .noiseless import NoiselessSolver
from .exhaustive import solve_exhaustive, POOLS, regional_history_law
from .mechanisms import relation_statistics, RELATION_PROBES, terminal_components, terminal_gap
from .paired import write_csv, paired_pools, AnnotationOracle, acquire, POOL_OFFSET

REFERENCE = Path(__file__).resolve().parents[1]/'reference'


def _read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def compare_printed_tables(out: Path):
    comparisons=[]
    for ref in _read(REFERENCE/'rq1_manuscript.csv'):
        d,b=int(ref['d']),int(ref['budget'])
        solver=NoiselessSolver(d); result=solver.solve_budget(b)
        if result['adaptive_cache_states']!=int(ref['adaptive_states']):
            raise AssertionError('RQ1 state count does not match manuscript')
        for method in ('uniform','fixed','latc','adaptive','basis'):
            value=result[method+'_unqueried']; expected=float(ref[method]); error=abs(value-expected)
            if error>5e-6+1e-12:
                raise AssertionError(f'RQ1 printed value mismatch: d={d}, method={method}')
            comparisons.append({'experiment':'rq1','setting':f'd={d},b={b}', 'method':method,
                                'computed':value,'printed_reference':expected,'absolute_error':error,
                                'rounding_tolerance':5e-6})
        solver.clear_caches()
    path=out/'rq3.csv'

    if path.exists():
        rows=_read(path)
    else:
        rows=[]
        for eta in (0.0,.05,.1,.2):
            for name,pool in POOLS.items():
                rows.extend(solve_exhaustive(4,pool,eta).rows(name))
    lookup={(str(row['pool']),float(row['eta']),int(row['budget'])):row for row in rows}
    for ref in _read(REFERENCE/'rq3_manuscript.csv'):
        key=(ref['pool'],float(ref['eta']),int(ref['budget']))
        result=lookup[key]
        for method in ('fixed','two_batch','adaptive'):
            value=float(result[method+'_unqueried']);expected=float(ref[method]);error=abs(value-expected)
            if error>5e-7+1e-12:
                raise AssertionError(f'RQ3 printed value mismatch: {key}, {method}')
            comparisons.append({'experiment':'rq3','setting':f'pool={key[0]},eta={key[1]},b={key[2]}',
                                'method':method,'computed':value,'printed_reference':expected,
                                'absolute_error':error,'rounding_tolerance':5e-7})
    write_csv(out/'validation_printed_tables.csv',comparisons)
    return {'rq1_values':10,'rq3_values':312,'all_match_at_printed_precision':True,
            'rq1_max_absolute_error_to_rounded_table':max(r['absolute_error'] for r in comparisons if r['experiment']=='rq1'),
            'rq3_max_absolute_error_to_rounded_table':max(r['absolute_error'] for r in comparisons if r['experiment']=='rq3'),
            'rq1_adaptive_states':{'CP8_b17':1588,'CP12_b21':4171}}


def validate_posteriors():
    rng=np.random.Generator(np.random.PCG64(20260917))
    maximum_sign=maximum_log=0.0
    count=0
    def compare(d,eta,xs,ys,eval_x=None):
        nonlocal maximum_sign,maximum_log,count
        fast=regional_posterior(d,eta,xs,ys)
        dense=dense_regional_posterior(d,eta,xs,ys,eval_x)
        fast_sign=fast.clean_sign if eval_x is None else fast.clean_sign[eval_x]
        sign_error=float(np.max(np.abs(fast_sign-dense.clean_sign)))
        if np.isfinite(fast.log_lr) and np.isfinite(dense.log_lr):
            log_error=abs(fast.log_lr-dense.log_lr)
        elif fast.log_lr==dense.log_lr:
            log_error=0.0
        else:
            raise AssertionError('zero reliable mass mismatch')
        maximum_sign=max(maximum_sign,sign_error);maximum_log=max(maximum_log,log_error)
        if sign_error>1e-10 or log_error>1e-10:
            raise AssertionError('transform and dense posterior disagree')
        count+=1
    for d in (1,2,3):
        for eta in (0.0,.05,.1,.25):
            for _ in range(10):
                n=int(rng.integers(0,(1 << d)+1))
                xs=rng.choice(1 << d,size=n,replace=False)
                ys=rng.integers(0,2,size=n)
                compare(d,eta,xs,ys)

    histories=0
    for d in (8,12):
        for eta in (.01,.025,.05):
            target=sample_target_array(d,eta,10**6*d+int(eta*10000))
            pools,_=paired_pools(d,target.seed+POOL_OFFSET)
            for budget in (int(1.5*(d+1)),int(1.9*(d+1))):
                for name in ('random','structured'):
                    oracle=AnnotationOracle(target.labels,pools[name],budget)
                    (qs,ys),_,_=acquire('prescribed_latc',d,eta,pools[name],oracle,budget)
                    coords=rng.choice(1 << d,size=24,replace=False)
                    for j in (0,1):
                        compare(d,eta,qs[j],ys[j],coords)
                    histories+=1
    return {'regional_posterior_comparisons':count,'main_scale_histories':histories,
            'max_clean_sign_error':maximum_sign,'max_log_likelihood_error':maximum_log,
            'tolerance':1e-10}


def validate_closed_forms():
    errors=[]
    for name,k in (('length_four',4),('length_eight',8)):
        for eta in (0.0,.05,.1,.2,.49):
            stat=relation_statistics(RELATION_PROBES[name],eta)
            error=abs(stat['testing_error']-(.5-(1-2*eta)**k/4))
            errors.append(error)
            if error>1e-12:
                raise AssertionError('unique-dependency testing law failed')
    for eta in (0.0,.01,.05,.1,.2,.25,.49):
        short=terminal_components(eta,4);long=terminal_components(eta,6)
        error=abs(short['risk_unqueried']-long['risk_unqueried']-terminal_gap(eta))
        errors.append(error)
        if error>1e-12:
            raise AssertionError('terminal risk gap identity failed')
    return {'identities':len(errors),'max_absolute_error':max(errors)}


def validate_rational(out: Path):
    from .rational import rational_optima
    rows=[]; maximum=0.0
    for eta in (Fraction(0),Fraction(1,20),Fraction(1,10),Fraction(1,5)):
        for name,pool,last_budget in (('full',(0,1,2,3),7),('basis',(0,1,2),6)):
            exact=rational_optima(2,pool,eta)
            numerical=solve_exhaustive(2,pool,float(eta))
            for budget in range(last_budget+1):
                for method in ('fixed','two_batch','adaptive'):
                    expected=exact[method][budget]
                    computed=float(getattr(numerical,method)[budget])
                    error=abs(float(expected)-computed);maximum=max(maximum,error)
                    if error>1e-12:
                        raise AssertionError('rational full-target and ternary-state optima disagree')
                    rows.append({'pool':name,'eta_fraction':str(eta),'budget':budget,'method':method,
                                 'full_risk_fraction':str(expected),'unqueried_risk_fraction':str(expected*Fraction(8,8-budget)),
                                 'computed_full':computed,'absolute_error':error})
    write_csv(out/'validation_rational.csv',rows)
    return {'policy_value_comparisons':len(rows),'max_absolute_error':maximum,'exact_arithmetic':'fractions.Fraction over complete target arrays'}


def validate_fourier_integer():
    likelihood_count=representation_count=0
    max_mass_error=0.0
    d=4;m=16;tasks=32
    full_clean=np.array([[(beta+((slope & x).bit_count() & 1))%2 for x in range(m)]
                         for beta in (0,1) for slope in range(m)],dtype=np.int8)
    signs=1-2*full_clean
    for pool in POOLS.values():
        pool=tuple(pool);k=len(pool)
        task_restrictions=[sum(int(y) << j for j,y in enumerate(row[list(pool)])) for row in full_clean]
        row_xors=[]
        for subset in range(1 << k):
            result=0
            for j,x in enumerate(pool):
                if subset & (1 << j):result^=(x << 1)|1
            row_xors.append(result)
        for eta in (Fraction(0),Fraction(1,20),Fraction(1,10),Fraction(1,5)):
            p,q=eta.numerator,eta.denominator
            mass,_,_=regional_history_law(d,pool,float(eta))
            for h in range(3**k):
                rem=h;observed_mask=observed_bits=0
                for j in range(k):
                    digit=rem%3;rem//=3
                    if digit:
                        observed_mask|=1 << j;observed_bits|=(digit-1) << j
                n=observed_mask.bit_count()
                weights=[]
                for clean in task_restrictions:
                    e=((clean^observed_bits)&observed_mask).bit_count()
                    weights.append((q-p)**(n-e)*p**e)
                weighted_sum=sum(weights)
                transform=[0]*(2*m)
                for subset in range(1 << k):
                    if subset & ~observed_mask:continue
                    weight=subset.bit_count()
                    term=(q-2*p)**weight*q**(n-weight)
                    if (subset & observed_bits).bit_count() & 1:term=-term
                    transform[row_xors[subset]]+=term
                if weighted_sum*(1 << n)!=tasks*transform[0]:
                    raise AssertionError('integer Fourier likelihood identity failed')
                likelihood_count+=1
                max_mass_error=max(max_mass_error,abs(mass[h]-weighted_sum/(tasks*q**n)))
                queried={pool[j] for j in range(k) if observed_mask & (1 << j)}
                for x in range(m):
                    if x in queried:continue
                    signed_sum=sum(int(signs[task,x])*weights[task] for task in range(tasks))
                    if signed_sum*(1 << n)!=tasks*transform[(x << 1)|1]:
                        raise AssertionError('integer target-representation identity failed')
                    representation_count+=1
    return {'likelihood_identities':likelihood_count,'target_representation_identities':representation_count,
            'all_cross_multiplied_integer_identities_exact':True,
            'max_float_probability_error':max_mass_error}


def rref_generators(n: int):

    for dimension in range(n+1):
        for pivots in combinations(range(n),dimension):
            nonpivots=[j for j in range(n) if j not in pivots]
            free=[(i,j) for i,pivot in enumerate(pivots) for j in nonpivots if j>pivot]
            for bits in range(1 << len(free)):
                rows=[1 << pivot for pivot in pivots]
                for position,(i,j) in enumerate(free):
                    if bits & (1 << position):rows[i]|=1 << j
                yield rows


def validate_spectrum(out: Path, maximum_n: int=7):
    counts=[];nonzero_total=spectrum_checks=zero_checks=0
    for n in range(1,maximum_n+1):
        total=nonzero=0
        weights=[s.bit_count() for s in range(1 << n)]
        for generator in rref_generators(n):
            total+=1;D=len(generator)
            if D==0:
                zero_checks+=5
                continue
            nonzero+=1
            codewords=[0]
            for row in generator:codewords += [v^row for v in codewords]
            contained=[0]*(1 << n)
            for word in codewords:contained[word]=1
            for j in range(n):
                for subset in range(1 << n):
                    if subset & (1 << j):contained[subset]+=contained[subset^(1 << j)]
            h=min(weights[subset]//(count.bit_length()-1)
                  for subset,count in enumerate(contained) if count>1)
            infosets=[subset for subset in range(1 << n) if weights[subset]==D and gf2_rank(row & subset for row in generator)==D]
            dp=[-1]*(1 << n);dp[0]=0
            for used in range(1 << n):
                if dp[used]<0:continue
                for info in infosets:
                    if not used & info:
                        dp[used|info]=max(dp[used|info],dp[used]+1)
            if max(dp)!=h:
                raise AssertionError('information-set packing mismatch')
            histogram=[0]*(n+1)
            for word in codewords:histogram[weights[word]]+=1
            for delta in (Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(1)):
                p,q=delta.numerator,delta.denominator
                left=sum(a*p**w*q**(n-w) for w,a in enumerate(histogram))
                right=(q**h+p**h)**D*q**(n-h*D)
                if left>right:raise AssertionError('integer spectrum inequality violated')
                spectrum_checks+=1
        nonzero_total+=nonzero
        counts.append({'code_length':n,'all_codes':total,'nonzero_codes':nonzero,'spectrum_checks':5*nonzero})
        print(f'spectrum validation: n={n}, all {total} codes passed',flush=True)
    write_csv(out/'validation_code_enumeration.csv',counts)
    return {'all_codes':sum(r['all_codes'] for r in counts),'nonzero_codes':nonzero_total,
            'integer_spectrum_inequalities':spectrum_checks,'zero_code_identities':zero_checks,
            'packing_mismatches':0,'spectrum_violations':0}


def validate(out: Path, deep: bool=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'printed_tables':compare_printed_tables(out),
            'posterior':validate_posteriors(),'closed_forms':validate_closed_forms()}
    if deep:
        report['rational_full_target']=validate_rational(out)
        report['fourier_integer']=validate_fourier_integer()
        report['binary_code_spectrum']=validate_spectrum(out)
    else:
        report['deep_checks']='not run; use --deep to include rational, integer Fourier, and all codes through length seven'
    report['status']='passed'
    (out/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
