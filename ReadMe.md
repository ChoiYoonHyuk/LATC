# Label-Access Geometry and the Value of Adaptivity in Noisy Affine Prediction

## Main experiment code

This package contains an independent Python implementation of the main experiments in **Label-Access Geometry and the Value of Adaptivity in Noisy Affine Prediction**. It is the previously provided implementation with Python comments and docstrings removed. Numerical algorithms, experimental settings, random-number generation, and test assertions are unchanged. Command-line descriptions remain ordinary runtime strings so that help output is preserved.

The archive contains source code, tests, dependency and project configuration files, small manuscript-derived regression fixtures, and this `ReadMe.md`. It does not contain generated experiment results, target archives, figures, execution logs, or previous reports. Those outputs are created when the commands below are executed.

The experiments use synthetic binary affine populations. No external dataset download, GPU, API key, or pretrained model is required. This implementation was written from the supplied manuscript; it is not a copy of the manuscript's linked repository.

## 1. Environment and installation

The project declares Python 3.10 or newer. The recorded execution environment uses Python 3.13.5 on Linux x86_64. `requirements.txt` contains dependency ranges, and `requirements-tested.txt` contains the recorded dependency versions. Compatibility with every Python version admitted by the project metadata has not been separately tested.

Extract the ZIP archive and open a terminal in the directory containing `run_experiments.py` and this file:

```bash
cd the_geometry_of_label_access_in_active_learning
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate the environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, activate it in Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Install dependencies using one of the following commands. The first uses the supported version ranges; the second uses the recorded versions.

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pip install -r requirements-tested.txt
```

Run commands from the extracted project directory. Installation with `pip install .` is not required for this workflow. Keep `reference/` alongside `latc/` because validation reads the regression fixtures from that location.

## 2. Run the main experiments

Run RQ1-RQ4 with 128 paired replicates per Monte Carlo setting:

```bash
python run_experiments.py all --replicates 128 --out results
```

This command also runs the standard validation suite and generates figures from the computed CSV files. All work is performed in the invoked process. Choose a new output directory for each independent execution to avoid mixing old and new outputs.

To include the matched-budget extension, public-prior sensitivity, and deeper independent validation:

```bash
python run_experiments.py all --replicates 128 --protocol both --public-priors --deep --out results_extended
```

### RQ1: Exact acquisition-class comparisons

```bash
python run_experiments.py rq1 --out results_rq1
```

RQ1 evaluates noiseless CP-8 and CP-12 across budgets from zero through twice the task dimension. It computes uniform-query risk, the full fixed optimum, LATC with optimized continuation, the full adaptive optimum, and the basis-pool optimum.

LATC in RQ1 optimizes the specified subcube-probe design family. It is not the optimum over every possible two-batch policy.

Add the public-prior sensitivity analysis:

```bash
python run_experiments.py rq1 --public-priors --out results_rq1_priors
```

### RQ2: Paired eligibility and completion budgets

```bash
python run_experiments.py rq2 --protocol window --replicates 128 --out results_window
```

The main window protocol uses dimensions 8 and 12, noise rates 0.01, 0.025, and 0.05, and budgets `floor(lambda * (d + 1))` for `lambda` in `(1.5, 1.7, 1.9)`. Each policy is evaluated on paired random and structured pools. The primary window metric is common-population Brier loss outside the union of both eligible pools.

The acquisition rules are Concentrated Fixed, Balanced Fixed, and Prescribed LATC. All acquired labels update the terminal posterior. Paired differences are defined as random-pool risk minus structured-pool risk, with pointwise 95% Student intervals.

Run the matched-budget extension:

```bash
python run_experiments.py rq2 --protocol matched --replicates 128 --out results_matched
```

This extension uses dimension 12, noise rates 0.05, 0.10, and 0.20, and budgets 32, 64, and 128. Its primary comparison uses unqueried Brier loss.

### RQ3: Exhaustive noisy acquisition optimization

```bash
python run_experiments.py rq3 --out results_rq3
```

RQ3 uses two dimension-four regions and six eligible vertices per region. The short-check pool is `(0, 1, 2, 3, 4, 8)` and the long-check pool is `(0, 1, 2, 4, 8, 15)`. The solver evaluates budgets 0-12 at noise rates 0, 0.05, 0.10, and 0.20.

Fixed, full two-batch, and fully adaptive policies are optimized independently over their complete finite classes. Equality between two-batch and adaptive values is checked after optimization, not assumed by the solver.

### RQ4: Verification mechanisms

```bash
python run_experiments.py rq4 --out results_rq4
```

RQ4 computes exact relation evidence, the analytic noise-scale binomial diagnostic, finite verification overhead, and the terminal task/reliability decomposition. Large values of `log2(r)` in the diagnostic are analytic calculations; the code does not allocate a population of size `2**r`.

## 3. Useful command-line options

| Option | Effect |
|---|---|
| `--out PATH` | Select the output directory. The default is `results`. |
| `--replicates N` | Set the number of RQ2 replicates. The default is 128. |
| `--protocol window` | Run the main RQ2 completion-window protocol. This is the default. |
| `--protocol matched` | Run the matched-budget extension. |
| `--protocol both` | Run both RQ2 protocols. |
| `--dimensions 8` | Restrict RQ1 and the RQ2 window protocol to dimension 8. |
| `--public-priors` | Add the RQ1 public-prior sensitivity grid. |
| `--no-save-targets` | Disable RQ2 target archive output. |
| `--skip-plots` | Skip figure generation. |
| `--deep` | Add deeper checks when the selected command includes validation. |

RQ3 remains at dimension 4. The matched-budget extension remains at dimension 12. The public-prior extension uses its fixed CP-8 and CP-12 operating points even when `--dimensions` restricts the main grid. `--replicates` does not reduce the exact RQ1 or RQ3 calculations, and `--deep` has an effect only with `all` or `validate`.

Show command-line help:

```bash
python run_experiments.py --help
python replay_saved.py --help
```

For a smaller installation check, run:

```bash
python run_experiments.py rq2 --protocol window --dimensions 8 --replicates 4 --out smoke_results
```

The reduced replicate count is for checking execution, not for the manuscript's 128-replicate evaluation. Use at least two replicates for sample standard errors and Student intervals.

Numba compiles the exhaustive numerical kernels on first use. RQ3 allocates arrays over `3**12` joint partial histories and processes `4**12` partial-set/final-set pairs per pool/noise setting. Deep validation additionally enumerates binary linear codes through length seven. These are exhaustive finite calculations, not large-scale graph-training benchmarks.

## 4. Tests and validation

Run the automated tests:

```bash
python -m pytest -q
```

Run only tests that are not marked as slow:

```bash
python -m pytest -q -m "not slow"
```

Run the standard validation command:

```bash
python run_experiments.py validate --out validation_results
```

Run deeper rational-arithmetic, integer Fourier, and binary-code spectrum checks:

```bash
python run_experiments.py validate --deep --out validation_deep_results
```

Validation uses manuscript-derived reference values in `reference/rq1_manuscript.csv` and `reference/rq3_manuscript.csv`. Acquisition algorithms do not read those values. The files are regression inputs, not newly generated experiment results.

Here, exact optimization means complete finite integration and optimization rather than Monte Carlo estimation or heuristic search. The main numerical solvers use `float64`; the independent small-population solver uses `fractions.Fraction`.

## 5. Generated outputs

Output names below are relative to the directory selected by `--out`. Each command produces only the outputs relevant to the stages it runs.

| Output | Contents |
|---|---|
| `rq1.csv` | Exact RQ1 risks, selected probes, allocations, and state counts. |
| `rq1_public_priors.csv` | Optional public-prior sensitivity results. |
| `window_runs.csv` | Per-replicate RQ2 window measurements. |
| `window_summary.csv` | Group means and standard errors. |
| `window_paired.csv` | Paired random-minus-structured differences and confidence intervals. |
| `window_mechanisms.csv` | Localization, task-energy, and rank diagnostics. |
| `matched_*.csv` | Corresponding matched-budget extension measurements. |
| `*_queries.jsonl.gz` | Acquired identities, labels, batches, and policy records. |
| `targets/*.npz` | Generated targets, latent evaluation metadata, public pools, and common masks. |
| `rq3.csv` | Exact fixed, two-batch, and adaptive values for every RQ3 configuration. |
| `rq3_terminal_components.csv` | Terminal risk and uncertainty components generated by the mechanisms stage. |
| `rq4_relations.csv` | Relation testing errors and discrepancies. |
| `rq4_noise_scale.csv` | Analytic noise-scale diagnostic. |
| `rq4_overhead.csv` | Finite verification overhead. |
| `validation_*.csv`, `validation_*.json` | Numerical and independent validation records. |
| `execution_<command>.json` | Command settings, environment information, and stage execution records. |
| `figures/*.png`, `figures/*.svg` | Figures generated from computed CSV files. |

Regenerate figures from an existing output directory:

```bash
python run_experiments.py plot --out results
```

The plotting command reads available CSV files; it does not rerun missing experiments.

## 6. Reproducibility and replay

Target arrays are generated once per identity and remain fixed during acquisition. Target generation uses `numpy.random.Generator(numpy.random.PCG64(seed))`. Public pools use a separate generator with a seed offset of `10**12`.

The window protocol uses:

```text
target_seed = 10**6 * d + 10**4 * rate_index + replicate_index
pool_seed = target_seed + 10**12
rate_order = (0.01, 0.025, 0.05)
```

Budget is absent from this seed, so the three budgets at a given dimension, rate, and replicate share targets, pools, and common evaluation populations.

The matched-budget extension uses:

```text
target_seed = 10**6 * rate_index + 1000 * budget + replicate_index
pool_seed = target_seed + 10**12
rate_order = (0.05, 0.10, 0.20)
```

RQ2 follows the specified experimental protocol, but bitwise equality with the manuscript's original Monte Carlo tables is not claimed. Integer seed formulas alone do not specify the original generator and complete random-call convention.

After running the window protocol with target saving enabled, replay an archived setting without generating new random numbers:

```bash
python replay_saved.py results/targets/window_d12_e0_all_budgets.npz --budget 24 --reference results/window_runs.csv --out replay.csv
```

Use the same output directory in this command that was used to generate the archive and reference CSV. Replay produces a CSV and a JSON comparison report. The latent task and reliable-region metadata in target archives are evaluator-only; policies acquire labels through the annotation oracle.

## 7. Source layout

```text
the_geometry_of_label_access_in_active_learning/
    ReadMe.md
    run_experiments.py
    replay_saved.py
    requirements.txt
    requirements-tested.txt
    pyproject.toml
    .gitignore
    latc/
        __init__.py
        affine.py
        cli.py
        exhaustive.py
        mechanisms.py
        noiseless.py
        paired.py
        plots.py
        rational.py
        validation.py
    reference/
        rq1_manuscript.csv
        rq3_manuscript.csv
        source.json
    tests/
        test_affine.py
        test_exact.py
        test_policies.py
```

`affine.py` handles binary affine tasks and posterior prediction. `noiseless.py`, `paired.py`, `exhaustive.py`, and `mechanisms.py` implement RQ1-RQ4. `rational.py` and `validation.py` provide independent checks. `plots.py` renders computed outputs, and `cli.py` coordinates execution.

## 8. Python API example

```python
from latc.noiseless import NoiselessSolver
from latc.exhaustive import POOLS, solve_exhaustive

solver = NoiselessSolver(d=12, prior=0.5)
try:
    result = solver.solve_budget(21)
    print(result["latc_unqueried"])
    print(result["adaptive_unqueried"])
finally:
    solver.clear_caches()

exact = solve_exhaustive(d=4, pool=POOLS["short"], eta=0.05)
print(exact.fixed)
print(exact.two_batch)
print(exact.adaptive)
```

The arrays returned by `solve_exhaustive` contain full-population risks indexed by total budget. For RQ3, multiply the value at budget `b` by `32 / (32 - b)` to obtain the unqueried normalization.

The package is limited to the previously implemented main experiments and listed extensions. It does not include Hadamard order-12 experiments, the correlated-dictionary stopped-bound grid, or fitting real-graph transfer-audit predictors.
