# LATC

**Query-pool geometry and Bayes risk in noisy affine prediction**

[한국어 안내](README.ko.md) · [Paper-to-code map](docs/PAPER_MAP.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Validation](docs/VALIDATION.md)

A Python implementation of the exclusive two-region affine prediction model,
exact Bayesian prediction, acquisition-class optimization, paired pool controls,
and finite risk bounds in the supplied 73-page ICLR 2027 submission manuscript.

**Provenance.** This is an independent implementation grounded in the supplied
manuscript, not a copy of the original authors' repository. No original source
code or per-replicate author data were supplied. The manuscript does **not** label
its experiments “RQ1–RQ4”; the four names below are repository-level organization
for this implementation. They are not quotations or newly attributed claims.

## Experiment groups

| Group | Implemented experiment | Manuscript basis |
|---|---|---|
| **RQ1** | Exact noiseless full-access Uniform, fixed, optimized subcube, full two-batch, and adaptive acquisition; batch-size profiles; basis-pool and public-prior controls | §6.1; Appendices G, H, K, L.1; Tables 4–8 |
| **RQ2** | Paired noisy prefix, two-check, subcube, and exhaustive affine-flat controls; persistent labels; common targets; conditional/realized risks; variance decomposition and paired inference | §6.2; Appendices L.2–L.5, P.1; Tables 9–17, 25–27 |
| **RQ3** | Exact noisy optimization on the matched six-point pools; common/population/unqueried risks; BALD, BatchBALD, EPIG, one-step Brier; tie controls and complete-acquisition score comparisons | Appendices M, N.1–N.5, O; Tables 18–24 |
| **RQ4** | Directed-rounding finite risk certificates; expected affine-flat counts; whole-pool short-dependency bounds; subcube verification at theorem-scale dimensions | §6.3; Appendices P.2–P.4, Q; Tables 28–36 |

The map in `docs/PAPER_MAP.md` links individual modules to equations and identifies
extensions that are outside this repository's scope.

## Installation

Use Python **3.10 or later**. The delivered artifact was tested on Python 3.13.5;
exact installed package versions are recorded in `requirements-tested.txt` and the
validation report. No GPU, dataset download, private service, or API key is needed.

```bash
cd LATC
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

A CPU-only scientific Python stack is used: NumPy, SciPy, and Matplotlib. The risk
certificate engine additionally uses Python's standard-library `decimal` module.
The paper's exponentially large populations are never materialized for RQ4.

## Quick start

Run the tests and four small, explicitly labeled smoke configurations:

```bash
python -m pytest -q
python scripts/run_all.py --preset smoke --out results/smoke --plots
```

`results/smoke` must be new or empty. Existing experiment output is never silently
overwritten. Re-running uses a different output directory. The delivered
`example_results/smoke/` contains actual executions, not manuscript numbers copied
into result files. Smoke Monte Carlo results are **not** paper-sized estimates.

Run each complete paper-parameter experiment separately:

```bash
latc run configs/rq1_paper.json --out results/rq1 --plots
latc run configs/rq2_paper.json --out results/rq2 --plots
latc run configs/rq3_paper.json --out results/rq3 --plots
latc run configs/rq4_paper.json --out results/rq4 --plots
```

Or execute all four in sequence:

```bash
python scripts/run_all.py --preset paper --out results/paper --plots
```

`python -m latc run ...` is equivalent to `latc run ...`. Remove `--plots` to produce
only numerical outputs. Configurations are JSON, and unknown top-level experiment
parameters raise an error instead of being ignored. Separate figures are written
as PNGs; their values come only from the computed CSVs.

For small array operations, a single BLAS thread can reduce overhead. On
Linux/macOS, for example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

This is an execution setting, not a change in the experiment or estimator.

## Minimal inference example

```python
from latc import predict_two_regions

prediction = predict_two_regions(
    d=8,
    coordinates=[[0, 1, 2, 3, 4, 5, 6, 7], []],
    labels=[[0, 1, 0, 1, 1, 0, 1, 0], []],
    eta=0.05,
    prior=0.5,
)

print(prediction.gamma)                 # P(region j is affine | history)
print(prediction.probabilities.shape)   # (2, 256)
print(prediction.population_risk)       # conditional Bayes Brier, all targets
print(prediction.unqueried_risk)        # conditional Bayes Brier, unqueried only
```

Regions are indexed **0 and 1 in Python and output files**, corresponding to
regions **1 and 2 in the manuscript**. Coordinates are integer encodings of binary
vectors. The affine parameter is encoded as `beta + 2*u`, and the augmented row
is `1 | (x << 1)`. Observed labels are persistent and are predicted exactly.

## What is reproduced

**Deterministic experiments.** RQ1 integrates every reduced-state outcome and
optimizes the full acquisition classes specified in Appendix H. RQ3 enumerates
every finite-pool action, first set, outcome, and fixed continuation. “Exact” means
complete finite integration, evaluated in floating-point arithmetic; it does not
mean every output is stored as a rational number. Regression fixtures contain
only the manuscript's rounded values and are never read by the algorithms.

**Monte Carlo experiments.** The paper presets retain seed 2026, the stated
sampling distributions, independent PCG64 streams, paired arrays, pool coupling,
noise rates, budgets, and sample sizes. The manuscript describes stream indexing
but does not provide its full integer encoding and random-draw implementation.
This repository therefore specifies a new, explicit stream map. Its results are
reproducible from this code, but **bit-for-bit identity with Tables 9–17, 25–27,
and 35 is not claimed**. Similar or different sample estimates must be interpreted
with their paired uncertainty, not forced to match the paper.

**Finite certificates.** RQ4 re-evaluates the finite sums and thresholds in the
manuscript. It compares outward-rounded numerical bounds before display rounding.
A strict result is a bound on the expected optimal risks under the two pool laws,
not a Monte Carlo estimate of a giant explicit population. The first certified
point in the direct preset is a point on the **evaluated grid**, not a proven
minimum dimension for separation.

## Paper presets and outputs

| Group | Full preset | Main output files |
|---|---|---|
| RQ1 | 46 CP-8/CP-12 budget rows; CP-8/12/16 basis curves; 21 public-prior cases | `curves.csv`, `batch_profiles.csv`, `basis_control.csv`, `public_priors.csv` |
| RQ2 | Independent 128-array, 512-array, and 512-array affine-flat studies for each of 6 dimension/noise pairs | `replicates.jsonl.gz`, `samples/*.npz`, `summaries.csv`, `paired_contrasts.csv`, `budget_interactions.csv`, `regional_strata.csv`, `first_batch_geometry.csv` |
| RQ3 | 104 pool/noise/budget configurations, each optimized on common and population targets; 416 public-order policy values plus sequential tie controls | `exact_curves.csv`, `acquisition_objectives.csv`, `complete_target_scores.csv`, `prior_formulas.csv` |
| RQ4 | 4 direct and 6 analytic certificate rows; 9 testing cells, each with 131072 observations **under each model** | `certificates.json`, `certificates.csv`, `flat_availability.csv`, `short_dependency_bounds.csv`, `verification.csv` |

Every output directory also contains `config.json` and `manifest.json`: the exact
configuration, configuration hash, dependency versions, timestamp, execution
status, and elapsed time. Numerical certificate endpoints are decimal **strings**
to preserve values that are far below binary floating-point display ranges.

## Scientific conventions that matter

The first batch in Algorithm 1 is the region-one designated block. Its labels
are charged to the shared budget. The threshold is **likelihood ratio one**, even
when the terminal predictor uses a nonuniform public prior. Ties within `1e-12`
choose region one. The entire continuation is committed before any continuation
outcome is used. Both regional designated blocks are excluded from their ordered
remainders, including the unqueried region-two block.

Two-check search inspects the complete **public coordinates**, not labels.
Affine-flat search is exhaustive through parallel affine planes and falls back
to two checks only when no eight-point flat is found. The exact likelihood keeps
all dependencies, including extra dependencies between selected checks.

Population Brier, unqueried Brier, and common-target Brier are distinct outputs.
The common targets lie outside **both** pool interventions and remain fixed across
policies and budgets within each replicate. Squared regional confidence and the
target label's own noise factor enter prediction. RQ2 records both integrated
conditional Brier and realized squared error on those same targets.

Paired intervals use independent **arrays**, not individual targets, policies, or
budgets, as sampling units. Budget interactions preserve within-array covariance
and use Bonferroni correction across the configured dimension/noise family. Flat
contrasts use a joint Holm family containing both conditional and realized risks.
The verification experiment uses two 97.5% Clopper–Pearson intervals, combined by
a union bound for at least 95% per-cell coverage; zero observed errors still have
a positive upper endpoint.

## Repository layout

```text
LATC/
├── README.md / README.ko.md
├── pyproject.toml
├── requirements.txt / requirements-tested.txt
├── configs/                    # smoke and paper presets for all four groups
├── src/latc/
│   ├── algebra.py              # GF(2), dependency bases, FWHT, syndrome DP
│   ├── posterior.py            # exact regional/mixture posterior and Brier risks
│   ├── pools.py / policies.py  # pool coupling, geometry search, Algorithm 1
│   ├── exact_noiseless.py      # sufficient-state acquisition optimization
│   ├── exact_noisy.py          # full ternary-history optimization and objectives
│   ├── comparisons.py          # matched-pool score formulas and finite sums
│   ├── intervals.py / bounds.py# directed-rounding finite certificates
│   ├── verification.py         # large-r, small-subcube testing simulation
│   ├── statistics.py / io.py / plotting.py / cli.py
│   └── experiments/rq1.py ... rq4.py
├── tests/                      # independent, structural, and paper regressions
├── reference/                  # explicitly labeled rounded manuscript fixtures
├── example_results/smoke/       # actual small execution artifacts
├── docs/                       # equation map, reproducibility, validation, release
├── scripts/run_all.py
└── .github/workflows/tests.yml
```

## Computational limits

Single-region exact prediction needs two Walsh–Hadamard transforms and scales as
`O(d * 2**d + n)` time and `O(2**d)` memory (Appendix I.1). Noiseless full access uses
count/rank/consistency states instead of population enumeration. General noisy
pool optimization has `3**K` partial histories: the supplied six-point pair has
`K=12` and 531441 states. The default noisy-search guard rejects larger problems;
it never silently replaces an optimum with a heuristic.

The RQ4 simulation operates only on subcubes of 64, 128, or 256 labels. It does
**not** decode an ambient parameter or simulate terminal population prediction at
`r=262144`; the terminal comparison there is supplied by finite risk bounds.

## Tests and release

```bash
python -m pytest -q                    # complete test suite, including regressions
python -m pytest -q -m "not slow"      # omit complete table/grid regression sweeps
```

`docs/VALIDATION.md` distinguishes the checks actually run for this delivery from
other tests that a maintainer may run later. A GitHub Actions workflow is included,
but no claim is made that a remote GitHub run has already occurred.

Before publishing, review `docs/RELEASE_CHECKLIST.md`. A license and final author
metadata have deliberately not been invented; add the maintainer-approved values.
The manuscript PDF is not redistributed in this repository.
