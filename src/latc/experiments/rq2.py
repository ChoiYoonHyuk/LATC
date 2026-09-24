"""RQ2 repository group: paired noisy public-pool interventions (L and P.1)."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import math
import re
import numpy as np
from ..algebra import augmented_rank, likelihood_table
from ..io import write_csv, write_jsonl_gzip
from ..pools import paired_pool_designs
from ..policies import fixed_control, generate_population, indexed_rng, prescribed_two_batch
from ..statistics import holm_adjust, student_summary


def _summarize(records: list[dict], out: Path) -> dict:
    metrics = ["conditional_common", "realized_common", "conditional_population", "realized_population",
               "conditional_unqueried", "parameter_precision", "residual", "parameter_uncertainty", "indicator_uncertainty"]
    groups = defaultdict(list)
    indexed = {}
    for row in records:
        key = tuple(row[k] for k in ("study", "d", "eta", "b", "policy"))
        groups[key].append(row)
        indexed[tuple(row[k] for k in ("study", "d", "eta", "b", "replicate", "policy"))] = row
    summaries = []
    for (study, d, eta, b, policy), rows in groups.items():
        for metric in metrics:
            summaries.append(dict(study=study, d=d, eta=eta, b=b, policy=policy, metric=metric,
                                  **student_summary([x[metric] for x in rows])))
        if rows[0]["selected_region"] is not None:
            summaries.append(dict(study=study, d=d, eta=eta, b=b, policy=policy, metric="selection_accuracy",
                                  **student_summary([x["selected_correctly"] for x in rows])))
    contrasts, deltas = [], {}
    comparisons = [("prefix", "two_check"), ("two_check", "subcube"), ("affine_flat", "subcube")]
    for (study, d, eta, b, policy), rows in groups.items():
        for left, right in comparisons:
            if policy != left or (study, d, eta, b, right) not in groups:
                continue
            paired = [(row, indexed[(study, d, eta, b, row["replicate"], right)])
                      for row in sorted(rows, key=lambda x: x["replicate"])]
            for metric in ("conditional_common", "realized_common"):
                diff = np.array([a[metric] - c[metric] for a, c in paired])
                item = dict(study=study, d=d, eta=eta, b=b, contrast=f"{left}_minus_{right}",
                            metric=metric, **student_summary(diff))
                contrasts.append(item)
                deltas[(study, d, eta, b, left, right, metric)] = diff
    # Joint Holm family contains conditional AND realized flat-minus-subcube tests,
    # across all dimensions, rates, and budgets within each independent study.
    for study in sorted({r["study"] for r in records}):
        selected = [row for row in contrasts if row["study"] == study and row["contrast"] == "affine_flat_minus_subcube"]
        if selected:
            for row, p in zip(selected, holm_adjust([row["p_value"] for row in selected])):
                row["holm_adjusted_p"] = float(p)
                row["holm_family_size"] = len(selected)
    interactions, strata = [], []
    experiment_groups = sorted({(r["study"], r["d"], r["eta"]) for r in records})
    for study in sorted({key[0] for key in experiment_groups}):
        valid_pairs = [(d, eta) for st, d, eta in experiment_groups if st == study
                       and any(key[:3] == (st, d, eta) and key[4:6] == ("two_check", "subcube") for key in deltas)]
        for d, eta in valid_pairs:
            budgets = sorted({r["b"] for r in records if (r["study"], r["d"], r["eta"]) == (study, d, eta)})
            if len(budgets) > 1:
                low, high = budgets[0], budgets[-1]
                key = lambda b: (study, d, eta, b, "two_check", "subcube", "conditional_common")
                interactions.append(dict(study=study, d=d, eta=eta, b_min=low, b_max=high,
                                         contrast="change_in_two_check_minus_subcube",
                                         **student_summary(deltas[key(high)] - deltas[key(low)], family_size=len(valid_pairs))))
            for b in budgets:
                rows = groups[(study, d, eta, b, "two_check")]
                for active in (0, 1):
                    diffs = [row["realized_common"] - indexed[(study, d, eta, b, row["replicate"], "subcube")]["realized_common"]
                             for row in rows if row["affine_region"] == active]
                    if len(diffs) >= 2:
                        strata.append(dict(study=study, d=d, eta=eta, b=b, affine_region=active,
                                           **student_summary(diffs)))
    write_csv(out / "summaries.csv", summaries)
    write_csv(out / "paired_contrasts.csv", contrasts)
    write_csv(out / "budget_interactions.csv", interactions)
    write_csv(out / "regional_strata.csv", strata)
    return dict(summary_rows=len(summaries), contrast_rows=len(contrasts), budget_interactions=len(interactions))


def run(parameters: dict, out: Path) -> dict:
    seed = int(parameters.get("seed", 2026))
    dimensions = parameters.get("dimensions", [8, 12])
    rates = parameters.get("noise_rates", [.01, .025, .05])
    lambdas = parameters.get("lambdas", [1.5, 1.7, 1.9])
    if len(set(dimensions)) != len(dimensions) or len(set(rates)) != len(rates):
        raise ValueError("Dimension and noise grids must not contain duplicate cells")
    prior = float(parameters.get("prior", .5))
    studies = parameters.get("studies", [dict(name="paired512", id=512, samples=512,
                                               flat_control=False, fixed_controls=False)])
    records, geometry = [], []
    save_samples = bool(parameters.get("save_samples", True))
    if save_samples:
        (out / "samples").mkdir(exist_ok=True)
    names = [s["name"] for s in studies]
    ids = [int(s["id"]) for s in studies]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Independent studies require unique names and stream IDs")
    for study in studies:
        if set(study) - {"name", "id", "samples", "flat_control", "fixed_controls"}:
            raise ValueError("Unknown nested study configuration key")
        if re.fullmatch(r"[A-Za-z0-9_-]+", str(study["name"])) is None:
            raise ValueError("Study names may contain only letters, digits, underscores, and hyphens")
        sample_count = int(study["samples"])
        if sample_count != study["samples"]:
            raise ValueError("Study sample count must be integral")
        if sample_count < 2:
            raise ValueError("Paired inference requires at least two independent arrays")
        for d in dimensions:
            budgets = [math.floor(float(ratio) * (d+1)) for ratio in lambdas]
            if len(set(budgets)) != len(budgets):
                raise ValueError("Budget ratios must produce distinct integer budgets")
            for eta in rates:
                saved = defaultdict(list)
                for replicate in range(sample_count):
                    target_rng = indexed_rng(seed, int(study["id"]), d, eta, replicate, 1)
                    pool_rng = indexed_rng(seed, int(study["id"]), d, eta, replicate, 2)
                    labels, affine_region, theta = generate_population(d, eta, target_rng, prior)
                    designs, targets = paired_pool_designs(d, pool_rng, include_flat=bool(study.get("flat_control", False)))
                    if save_samples:
                        saved["labels"].append(labels)
                        saved["affine_region"].append(affine_region)
                        saved["parameter"].append(theta)
                        saved["random_pools"].append(np.stack(designs["prefix"].pools))
                        saved["structured_pools"].append(np.stack(designs["subcube"].pools))
                        for name, design in designs.items():
                            saved[f"blocks_{name}"].append(np.stack(design.blocks))
                    for policy, design in designs.items():
                        for j, block in enumerate(design.blocks):
                            table = likelihood_table(block, eta)
                            geometry.append(dict(study=study["name"], d=d, eta=eta, replicate=replicate,
                                                 policy=policy, region=j, first_rank=augmented_rank(block),
                                                 flat_available=design.flat_available[j],
                                                 exact_balanced_accuracy=0.5+0.25*float(np.mean(abs(table-1)))))
                    for budget in budgets:
                        actions = [(name, design, None) for name, design in designs.items()]
                        if study.get("fixed_controls", False):
                            for name in ("prefix", "subcube"):
                                for balanced in (False, True):
                                    policy = f"{name}_{'balanced' if balanced else 'concentrated'}"
                                    actions.append((policy, designs[name], balanced))
                        for policy, design, balanced in actions:
                            if balanced is None:
                                acquired = prescribed_two_batch(d, eta, prior, design, labels, budget)
                            else:
                                acquired = fixed_control(d, eta, prior, design, labels, budget, balanced)
                            prediction = acquired.prediction
                            decomposition = prediction.decomposition(targets)
                            true_posterior = prediction.regional[affine_region]
                            records.append(dict(study=study["name"], study_id=int(study["id"]),
                                d=d, r=d+1, eta=eta, prior=prior, b=budget, replicate=replicate, policy=policy,
                                affine_region=affine_region, parameter=theta,
                                selected_region=acquired.selected_region,
                                selected_correctly=(int(acquired.selected_region == affine_region)
                                                    if acquired.selected_region is not None else None),
                                first_likelihood=acquired.first_likelihood,
                                gamma1=float(prediction.gamma[0]),
                                conditional_common=decomposition["conditional_brier"],
                                realized_common=prediction.realized_risk(labels, targets),
                                conditional_population=prediction.population_risk,
                                realized_population=prediction.realized_risk(labels),
                                conditional_unqueried=prediction.unqueried_risk,
                                residual=decomposition["residual"],
                                parameter_uncertainty=decomposition["parameter_uncertainty"],
                                indicator_uncertainty=decomposition["indicator_uncertainty"],
                                parameter_precision=float(np.mean(true_posterior.sign_mean**2)),
                                rank_affine=augmented_rank(acquired.queries[affine_region]),
                                rank_first=augmented_rank(design.blocks[0]) if balanced is None else None,
                                common_target_counts=[len(t) for t in targets],
                                queries=[q.tolist() for q in acquired.queries],
                                observed_labels=[y.tolist() for y in acquired.labels]))
                if save_samples:
                    filename = f"{study['name']}_d{d}_eta{int(round(eta*1e6)):06d}.npz"
                    np.savez_compressed(out / "samples" / filename,
                                        **{k: np.asarray(v) for k, v in saved.items()})
    write_jsonl_gzip(out / "replicates.jsonl.gz", records)
    write_csv(out / "first_batch_geometry.csv", geometry)
    result = _summarize(records, out)
    result.update(replicate_policy_budget_rows=len(records), geometry_rows=len(geometry), independent_studies=len(studies))
    return result
