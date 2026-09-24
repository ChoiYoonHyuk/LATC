"""RQ4 repository group: interval-certified finite risks and block verification."""
from __future__ import annotations
from pathlib import Path
from ..bounds import finite_certificate, expected_affine_flats, short_dependency_bound
from ..intervals import IntervalArithmetic
from ..io import write_csv, write_json
from ..verification import simulate_verification


def run(parameters: dict, out: Path) -> dict:
    precision = int(parameters.get("precision",80))
    certificates = []
    for method, default in [("direct",[768,1024,2048,4096]),("analytic",[256,1024,4096,16384,65536,262144])]:
        for r in parameters.get(f"{method}_dimensions",default):
            certificates.append(finite_certificate(r,method,precision))
    write_json(out / "certificates.json", certificates)
    write_csv(out / "certificates.csv", certificates)
    flats, arithmetic = [], IntervalArithmetic(precision)
    for d in parameters.get("flat_dimensions",[8,12,16,20,24,32]):
        expected = expected_affine_flats(d)
        interval = arithmetic.number(expected.numerator) / expected.denominator
        flats.append(dict(d=d,r=d+1,q=(d+1)**2,expected_flats=float(expected),
                          expectation_lower=str(interval.lo),expectation_upper=str(interval.hi),
                          existence_probability_upper=str(min(interval.hi,arithmetic.number(1).hi))))
    write_csv(out / "flat_availability.csv",flats)
    verification_dimensions = parameters.get("verification_dimensions",[1024,4096,262144])
    short = [short_dependency_bound(r,precision) for r in verification_dimensions]
    write_csv(out / "short_dependency_bounds.csv",short)
    testing = []
    if parameters.get("simulate_verification",True):
        for r in verification_dimensions:
            for eta in parameters.get("verification_noise_rates",[.05,.10,.20]):
                testing.append(simulate_verification(r,eta,int(parameters.get("verification_samples",131072)),
                    int(parameters.get("seed",2026)),int(parameters.get("chunk_size",512)),
                    int(parameters.get("verification_study_id",4096))))
    write_csv(out / "verification.csv",testing)
    return dict(certificate_rows=len(certificates),strict_certificates=sum(row["strict_certificate"] for row in certificates),
                flat_rows=len(flats),verification_rows=len(testing))
