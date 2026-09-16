from __future__ import annotations
import argparse
from pathlib import Path
import json
import platform
import time
import importlib.metadata
from .paired import write_csv, run_paired
from .noiseless import NoiselessSolver
from .exhaustive import solve_exhaustive, POOLS
from .mechanisms import run_mechanisms


def run_rq1(out: Path, dimensions=(8,12), public_priors: bool=False):
    rows=[]
    for d in dimensions:
        for budget in range(min(2*(d+1), 2*(1 << d)-1)+1):

            solver=NoiselessSolver(d)
            rows.append(solver.solve_budget(budget))
            solver.clear_caches()
        print(f"RQ1: CP-{d}, all budgets complete", flush=True)
    write_csv(out/'rq1.csv',rows)
    if public_priors:
        priors=[]
        for d,b in ((8,17),(12,21)):
            for p in (.5,.6,.7,.8,.9,.95,.99):
                solver=NoiselessSolver(d,p)
                priors.append(solver.solve_budget(b));solver.clear_caches()
        write_csv(out/'rq1_public_priors.csv',priors)
    return {'configurations':len(rows),'method':'exact expectation, complete fixed/adaptive classes; restricted LATC family'}


def run_rq3(out: Path):
    rows=[]
    for eta in (0.0,.05,.10,.20):
        for name,pool in POOLS.items():
            result=solve_exhaustive(4,pool,eta)
            rows.extend(result.rows(name))
            print(f"RQ3: {name}, eta={eta:g}, all budgets complete",flush=True)
    write_csv(out/'rq3.csv',rows)
    return {'configurations':len(rows),'acquisition_values':3*len(rows),
            'joint_states_per_pool_noise':3**12,'marginal_pairs_per_pool_noise':4**12,
            'two_batch_computed_independently':True}


def environment():
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','numba')}
    return {'python':platform.python_version(),'platform':platform.platform(),'dependencies':versions}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Command-line entry point. All work runs synchronously in this process.')
    parser.add_argument('experiment',choices=('all','rq1','rq2','rq3','rq4','validate','plot'))
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--replicates',type=int,default=128)
    parser.add_argument('--protocol',choices=('window','matched','both'),default='window')
    parser.add_argument('--dimensions',type=int,nargs='+',default=[8,12])
    parser.add_argument('--public-priors',action='store_true')
    parser.add_argument('--save-targets',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--skip-plots',action='store_true')
    parser.add_argument('--deep',action='store_true',help='also run rational full-target, Fourier, and binary-code validation')
    args=parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    manifest={'command':vars(args)|{'out':str(args.out)},'environment':environment(),'stages':{}}
    def stage(name,callback):
        start=time.perf_counter(); report=callback()
        manifest['stages'][name]={'elapsed_seconds':time.perf_counter()-start,'report':report}
    which=args.experiment
    if which in ('all','rq1'):
        stage('rq1',lambda:run_rq1(args.out,args.dimensions,args.public_priors))
    if which in ('all','rq2'):
        protocols=('window','matched') if args.protocol=='both' else (args.protocol,)
        for protocol in protocols:
            stage(protocol,lambda p=protocol:run_paired(args.out,p,args.replicates,args.dimensions,args.save_targets))
    if which in ('all','rq3'):
        stage('rq3',lambda:run_rq3(args.out))
    if which in ('all','rq4'):
        stage('rq4',lambda:run_mechanisms(args.out))
    if which in ('all','validate'):
        from .validation import validate
        stage('validation',lambda:validate(args.out,deep=args.deep))
    if not args.skip_plots and which not in ('validate',):
        from .plots import plot_outputs
        stage('plots',lambda:{'files':plot_outputs(args.out)})
    manifest['elapsed_seconds']=time.perf_counter()-started
    (args.out/f'execution_{which}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
