"""RQ1 repository group: noiseless full-access rounds, budgets, and public priors."""
from __future__ import annotations
from pathlib import Path
from ..exact_noiseless import NoiselessOptimizer, basis_risk
from ..io import write_csv


def run(parameters: dict, out: Path) -> dict:
    dimensions = parameters.get("dimensions", [8, 12])
    configured_budgets = parameters.get("budgets", "all")
    prior = float(parameters.get("prior", 0.5))
    curves, profiles = [], []
    for d in dimensions:
        solver = NoiselessOptimizer(d, prior)
        budgets = range(2*(d+1)+1) if configured_budgets == "all" else configured_budgets
        for b in budgets:
            profile = solver.two_batch_profile(b)
            sub = solver.subcube(b)
            row = dict(d=d, r=d+1, b=b, prior=prior, eta=0.0,
                       uniform=solver.uniform(b), fixed=solver.fixed(b), subcube=sub[0],
                       two_batch=solver.normalize(min(x["population_risk"] for x in profile), b),
                       adaptive=solver.adaptive(b), subcube_first_plan=sub[1], normalization="unqueried")
            if not row["adaptive"] <= row["two_batch"] + 2e-11 or not row["two_batch"] <= row["subcube"] + 2e-11 or not row["subcube"] <= row["fixed"] + 2e-11:
                raise ArithmeticError("Noiseless acquisition-class ordering failed")
            curves.append(row)
            for p in profile:
                profiles.append(dict(d=d, b=b, prior=prior, first_batch=p["first_batch"],
                                     risk=solver.normalize(p["population_risk"], b), first_plan=p["first_plan"]))
    basis = [dict(d=d, b=b, eta=float(eta), prior=prior,
                  all_classes=basis_risk(d, b, float(eta), prior), normalization="unqueried")
             for d in parameters.get("basis_dimensions", [8, 12, 16])
             for eta in parameters.get("basis_noise_rates", [0.0]) for b in range(2*(d+1)+1)]
    priors = []
    for d in parameters.get("prior_dimensions", []):
        b = d + 9  # Table 8: CP-8/12/16 at budgets 17/21/25.
        for p in parameters.get("prior_values", [.5, .6, .7, .8, .9, .95, .99]):
            solver = NoiselessOptimizer(d, p)
            priors.append(dict(d=d, b=b, prior=p, fixed=solver.fixed(b), subcube=solver.subcube(b)[0],
                               two_batch=solver.two_batch(b), adaptive=solver.adaptive(b)))
    write_csv(out / "curves.csv", curves)
    write_csv(out / "batch_profiles.csv", profiles)
    write_csv(out / "basis_control.csv", basis)
    write_csv(out / "public_priors.csv", priors)
    return dict(curve_rows=len(curves), batch_profile_rows=len(profiles), basis_rows=len(basis), prior_rows=len(priors))
