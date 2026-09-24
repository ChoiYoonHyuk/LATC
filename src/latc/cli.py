"""Command-line experiment execution with explicit JSON configurations."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import time
from .io import manifest, write_json

ALLOWED = {
    "rq1": {"dimensions","budgets","prior","basis_dimensions","basis_noise_rates","prior_dimensions","prior_values"},
    "rq2": {"seed","dimensions","noise_rates","lambdas","prior","studies","save_samples"},
    "rq3": {"noise_rates","budgets","prior","check_lengths","metrics","objectives","tie_conventions","formula_priors"},
    "rq4": {"precision","direct_dimensions","analytic_dimensions","flat_dimensions","verification_dimensions",
            "simulate_verification","verification_noise_rates","verification_samples","seed","chunk_size","verification_study_id"},
}


def execute(config_path: Path, out: Path, plots: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(config) - {"rq","description","parameters"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    rq = config.get("rq")
    if rq not in ALLOWED:
        raise ValueError("Configuration must select rq1, rq2, rq3, or rq4")
    parameters = config.get("parameters",{})
    unknown = set(parameters) - ALLOWED[rq]
    if unknown:
        raise ValueError(f"Unknown {rq} parameter keys: {sorted(unknown)}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output path: {out}. Choose a new directory.")
    out.mkdir(parents=True,exist_ok=True)
    metadata = manifest(config,config_path)
    metadata["status"] = "running"
    write_json(out/"manifest.json",metadata)
    write_json(out/"config.json",config)
    start = time.perf_counter()
    try:
        runner=importlib.import_module(f"latc.experiments.{rq}")
        result=runner.run(parameters,out)
        if plots:
            from .plotting import create_plots
            create_plots(rq,out)
    except Exception as error:
        metadata.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_seconds=time.perf_counter()-start)
        write_json(out/"manifest.json",metadata)
        raise
    metadata.update(status="complete",elapsed_seconds=time.perf_counter()-start,result=result,plots=plots)
    write_json(out/"manifest.json",metadata)
    return metadata


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog="latc",description="Paper-grounded LATC experiment runner")
    parser.add_argument("--version",action="version",version="LATC 0.1.0")
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run",help="Execute a JSON experiment configuration")
    run.add_argument("config",type=Path)
    run.add_argument("--out",type=Path,required=True,help="New/empty output directory; existing results are never overwritten")
    run.add_argument("--plots",action="store_true",help="Generate separate PNG figures from computed results")
    arguments=parser.parse_args(argv)
    try:
        result=execute(arguments.config,arguments.out,arguments.plots)
    except (ValueError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,f"latc: {error}\n")
    print(json.dumps({"output":str(arguments.out),"elapsed_seconds":result["elapsed_seconds"],**result["result"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
