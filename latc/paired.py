from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import numpy as np
from scipy.stats import t as student_t
from .affine import sample_target_array, regional_posterior, posterior_prediction

POLICIES = ("concentrated", "balanced", "prescribed_latc")
WINDOW_RATES = (0.01, 0.025, 0.05)
MATCHED_RATES = (0.05, 0.10, 0.20)
POOL_OFFSET = 10**12


@dataclass(frozen=True)
class EligiblePool:
    probe: np.ndarray
    reservoir: np.ndarray

    @property
    def coordinates(self):
        return np.concatenate((self.probe, self.reservoir))


def probe_size(d: int) -> int:
    return 1 << math.ceil(math.log2(math.log(d+1)**2))


def paired_pools(d: int, seed: int, t: int | None = None):
    m, q = 1 << d, (d+1)**2
    t = probe_size(d) if t is None else int(t)
    if t < 1 or t & (t-1) or not t <= q <= m:
        raise ValueError("need a power-of-two probe size with t <= (d+1)**2 <= 2**d")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    random, structured, common = [], [], []
    for _ in range(2):
        order = rng.permutation(m)
        rp = EligiblePool(order[:t].copy(), order[t:q].copy())
        sp = EligiblePool(np.arange(t, dtype=np.int64), order[order >= t][:q-t].copy())
        keep = np.ones(m, dtype=bool)
        keep[rp.coordinates] = False
        keep[sp.coordinates] = False
        random.append(rp)
        structured.append(sp)
        common.append(keep)
    if sum(int(x.sum()) for x in common) == 0:
        raise ValueError("common evaluation population is empty")
    return {"random": tuple(random), "structured": tuple(structured)}, np.stack(common)


class AnnotationOracle:

    def __init__(self, targets, pools: tuple[EligiblePool, EligiblePool], budget: int):
        self._labels = targets
        self._eligible = [set(map(int, p.coordinates)) for p in pools]
        self.budget = int(budget)
        self._seen: set[tuple[int, int]] = set()
        self.history: list[dict] = []
        self.batches = 0

    def query_batch(self, requests):
        requests = [(int(j), int(x)) for j, x in requests]
        if len(set(requests)) != len(requests) or any(key in self._seen for key in requests):
            raise ValueError("duplicate query: every fixed identity is charged only once")
        if len(self.history)+len(requests) > self.budget:
            raise ValueError("annotation budget exceeded")
        if any(j not in (0, 1) or x not in self._eligible[j] for j, x in requests):
            raise ValueError("ineligible identity")

        if requests:
            self.batches += 1
        answers = []
        for j, x in requests:
            y = int(self._labels[j, x])
            self._seen.add((j, x))
            self.history.append({"region": j, "coordinate": x, "label": y, "batch": self.batches})
            answers.append(y)
        return np.array(answers, dtype=np.int8)

    def observations(self):
        queries, labels = [], []
        for j in (0, 1):
            rows = [row for row in self.history if row["region"] == j]
            queries.append(np.array([row["coordinate"] for row in rows], dtype=np.int64))
            labels.append(np.array([row["label"] for row in rows], dtype=np.int8))
        return tuple(queries), tuple(labels)


def acquire(policy: str, d: int, eta: float, pools, oracle: AnnotationOracle,
            budget: int):

    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if budget != oracle.budget or budget < 0:
        raise ValueError("invalid budget")
    selected = None
    log_lr = None
    if policy == "prescribed_latc":
        probe = pools[0].probe
        if len(probe) > budget:
            raise ValueError("budget is smaller than the prescribed probe")
        observed = oracle.query_batch((0, x) for x in probe)
        post = regional_posterior(d, eta, probe, observed)
        log_lr = post.log_lr

        selected = 0 if log_lr >= 0 else 1
        remaining = budget-len(probe)
        if remaining > len(pools[selected].reservoir):
            raise ValueError("selected reservoir cannot hold the completion batch")
        oracle.query_batch((selected, x) for x in pools[selected].reservoir[:remaining])
    else:
        counts = (budget, 0) if policy == "concentrated" else (budget//2, budget-budget//2)
        if any(counts[j] > len(pools[j].reservoir) for j in (0, 1)):
            raise ValueError("fixed allocation exceeds reservoir capacity")
        oracle.query_batch((j, x) for j in (0, 1) for x in pools[j].reservoir[:counts[j]])
    if len(oracle.history) != budget:
        raise AssertionError("policy did not acquire exactly b distinct labels")
    if oracle.batches > (2 if policy == "prescribed_latc" else 1):
        raise AssertionError("too many acquisition batches")
    return oracle.observations(), selected, log_lr


def evaluate(target, pools, common_mask, policy: str, budget: int):
    oracle = AnnotationOracle(target.labels, pools, budget)
    (queries, observed), selected, probe_log_lr = acquire(policy, target.d, target.eta, pools, oracle, budget)
    probabilities, posts, gamma = posterior_prediction(target.d, target.eta, queries, observed)
    squared_error = (probabilities-target.labels)**2
    population = target.labels.size
    if budget >= population:
        raise ValueError("unqueried risk is undefined when no target remains")
    full = float(squared_error.mean())
    unqueried = float(squared_error.sum()/(population-budget))
    common = float(squared_error[common_mask].mean())
    for j in (0, 1):
        if np.any(common_mask[j, queries[j]]):
            raise AssertionError("evaluation target leaked into acquired labels")
        if not np.array_equal(probabilities[j, queries[j]], observed[j]):
            raise AssertionError("queried targets were not predicted exactly")
    reliable = target.reliable_region
    row = {"full": full, "unqueried": unqueried, "common": common,
           "selected_region": selected, "probe_log_lr": probe_log_lr,
           "localization_correct": int(selected == reliable) if selected is not None else None,
           "task_energy": float(np.mean(posts[reliable].clean_sign**2)),
           "reliable_query_rank": posts[reliable].rank,
           "posterior_reliable_probability": float(gamma[reliable]),
           "regional_counts": [len(q) for q in queries],
           "regional_ranks": [p.rank for p in posts],
           "posterior_gamma": list(map(float, gamma)),
           "batches": oracle.batches,
           "common_population_size": int(common_mask.sum())}
    return row, oracle.history


def write_csv(path: Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    return float(values.mean()), float(values.std(ddof=1)/math.sqrt(n)) if n > 1 else math.nan


def summarize_records(records: list[dict]):
    groups = defaultdict(list)
    for row in records:
        groups[(row["protocol"], row["d"], row["eta"], row["budget"], row["policy"], row["pool"])].append(row)
    summaries = []
    for key in sorted(groups):
        values = groups[key]
        base = dict(zip(("protocol", "d", "eta", "budget", "policy", "pool"), key))
        for metric in ("common", "unqueried", "full"):
            mean, se = mean_se([v[metric] for v in values])
            summaries.append({**base, "metric": metric, "n": len(values), "mean": mean, "se": se})
    comparisons = []
    contrast_keys = sorted(set(key[:-1] for key in groups))
    for key in contrast_keys:
        random = {row["replicate"]: row for row in groups[(*key, "random")]}
        structured = {row["replicate"]: row for row in groups[(*key, "structured")]}
        if random.keys() != structured.keys():
            raise AssertionError("unpaired replicates")
        for i in random:
            if random[i]["target_seed"] != structured[i]["target_seed"]:
                raise AssertionError("contrasts do not share a target array")
            if random[i]["common_hash"] != structured[i]["common_hash"]:
                raise AssertionError("contrasts do not share common evaluation targets")
        for metric in ("common", "unqueried", "full"):
            differences = [random[i][metric]-structured[i][metric] for i in sorted(random)]
            mean, se = mean_se(differences)
            n = len(differences)
            halfwidth = float(student_t.ppf(0.975, n-1)*se) if n > 1 else math.nan
            comparisons.append({**dict(zip(("protocol", "d", "eta", "budget", "policy"), key)),
                                "metric": metric, "n": n, "difference": mean, "se": se,
                                "ci95_halfwidth": halfwidth, "ci95_low": mean-halfwidth,
                                "ci95_high": mean+halfwidth})
    mechanisms = []
    for key in sorted(groups):
        if key[-2] != "prescribed_latc":
            continue
        values = groups[key]
        mean_e, se_e = mean_se([v["task_energy"] for v in values])
        mechanisms.append({**dict(zip(("protocol", "d", "eta", "budget", "policy", "pool"), key)),
                           "n": len(values), "localization_accuracy": float(np.mean([v["localization_correct"] for v in values])),
                           "task_energy": mean_e, "task_energy_se": se_e,
                           "mean_reliable_query_rank": float(np.mean([v["reliable_query_rank"] for v in values]))})
    return summaries, comparisons, mechanisms


def _save_targets(path: Path, targets, public_pools, common_masks):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        labels=np.stack([t.labels for t in targets]),
        reliable_regions=np.array([t.reliable_region for t in targets]),
        beta=np.array([t.beta for t in targets]), slopes=np.array([t.slope for t in targets]),
        target_seeds=np.array([t.seed for t in targets], dtype=np.int64),
        pool_seeds=np.array([t.seed+POOL_OFFSET for t in targets], dtype=np.int64),
        d=np.array(targets[0].d), eta=np.array(targets[0].eta),
        random_pools=np.stack([[p.coordinates for p in pools["random"]] for pools in public_pools]),
        structured_pools=np.stack([[p.coordinates for p in pools["structured"]] for pools in public_pools]),
        common_masks=np.stack(common_masks))


def run_paired(out: Path, protocol: str = "window", replicates: int = 128,
               dimensions=(8, 12), save_targets: bool = True):

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if protocol not in ("window", "matched"):
        raise ValueError("protocol must be 'window' or 'matched'")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rates = WINDOW_RATES if protocol == "window" else MATCHED_RATES
    dims = dimensions if protocol == "window" else (12,)
    records = []
    target_count = 0
    with gzip.open(out/f"{protocol}_queries.jsonl.gz", "wt", encoding="utf-8") as log:
        for d in dims:
            budgets = tuple(math.floor(lam*(d+1)) for lam in (1.5, 1.7, 1.9)) if protocol == "window" else (32, 64, 128)
            for e, eta in enumerate(rates):
                groups = [budgets] if protocol == "window" else [(b,) for b in budgets]
                for budget_group in groups:
                    target_group, pools_group, common_group = [], [], []
                    for i in range(replicates):
                        seed = 10**6*d+10**4*e+i if protocol == "window" else 10**6*e+1000*budget_group[0]+i
                        target = sample_target_array(d, eta, seed)
                        pools, common_mask = paired_pools(d, seed+POOL_OFFSET, t=8 if protocol == "matched" else None)
                        common_hash = hashlib.sha256(np.packbits(common_mask).tobytes()).hexdigest()
                        target_hash = hashlib.sha256(target.labels.tobytes()).hexdigest()
                        target_count += 1
                        if save_targets:
                            target_group.append(target); pools_group.append(pools); common_group.append(common_mask)
                        for budget in budget_group:
                            for policy in POLICIES:
                                for pool_name in ("random", "structured"):
                                    row, history = evaluate(target, pools[pool_name], common_mask, policy, budget)
                                    row.update({"protocol": protocol, "d": d, "eta": eta, "budget": budget,
                                                "replicate": i, "policy": policy, "pool": pool_name,
                                                "target_seed": seed, "pool_seed": seed+POOL_OFFSET,
                                                "common_hash": common_hash, "target_hash": target_hash})
                                    records.append(row)
                                    log.write(json.dumps({**row, "queries": history}, allow_nan=False)+"\n")
                    if save_targets:
                        group_label = "all_budgets" if protocol == "window" else f"b{budget_group[0]}"
                        _save_targets(out/"targets"/f"{protocol}_d{d}_e{e}_{group_label}.npz", target_group, pools_group, common_group)
                    print(f"{protocol}: d={d}, eta={eta:g}, budgets={budget_group}, arrays={replicates}", flush=True)
    summaries, comparisons, mechanisms = summarize_records(records)
    write_csv(out/f"{protocol}_runs.csv", records)
    write_csv(out/f"{protocol}_summary.csv", summaries)
    write_csv(out/f"{protocol}_paired.csv", comparisons)
    write_csv(out/f"{protocol}_mechanisms.csv", mechanisms)
    report = {"protocol": protocol, "independent_target_arrays": target_count,
              "prediction_records": len(records), "replicates_per_setting": replicates,
              "exact_budget_distinctness_eligibility_checks": "passed for every record",
              "common_evaluation_disjointness": "passed for every record",
              "seed_policy": "manuscript integers; explicit independent PCG64 call convention",
              "original_monte_carlo_table_bitwise_match_claimed": False}
    (out/f"{protocol}_execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
