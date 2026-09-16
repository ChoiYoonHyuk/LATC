from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_table(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory/f"{name}.png", dpi=180)
    fig.savefig(directory/f"{name}.svg")
    plt.close(fig)


def plot_outputs(out: Path):
    out = Path(out)
    figures = out/"figures"
    made = []
    path = out/"rq1.csv"
    if path.exists():
        rows = read_table(path)
        for d in sorted({int(r["d"]) for r in rows}):
            vals = sorted((r for r in rows if int(r["d"]) == d and float(r["prior"]) == 0.5), key=lambda r: int(r["budget"]))
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for column, label, line in [("uniform", "Uniform queries", ":"), ("fixed", "Optimal fixed", "--"),
                                       ("latc", "LATC, optimized continuation", "-"),
                                       ("adaptive", "Optimal adaptive", "-."), ("basis", "Basis: all-class optimum", "--")]:
                ax.plot([int(r["budget"]) for r in vals], [float(r[column+"_unqueried"]) for r in vals],
                        linestyle=line, label=label, linewidth=1.8)
            ax.set(xlabel="Charged label budget b", ylabel="Exact unqueried Brier risk", title=f"RQ1 | CubePair-{d}")
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
            name = f"rq1_cp{d}"; _save(fig, figures, name); made.append(name)
    for protocol, metric in (("window", "common"), ("matched", "unqueried")):
        path = out/f"{protocol}_paired.csv"
        if not path.exists():
            continue
        rows = [r for r in read_table(path) if r["metric"] == metric and r["policy"] == "prescribed_latc"]
        for d in sorted({int(r["d"]) for r in rows}):
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for eta in sorted({float(r["eta"]) for r in rows}):
                vals = sorted((r for r in rows if int(r["d"]) == d and float(r["eta"]) == eta), key=lambda r: int(r["budget"]))
                ax.errorbar([int(r["budget"]) for r in vals], [float(r["difference"]) for r in vals],
                            yerr=[float(r["ci95_halfwidth"]) for r in vals],
                            marker="o", capsize=3, label=f"eta={eta:g}")
            ax.axhline(0, linestyle=":", linewidth=1)
            ax.set(xlabel="Charged label budget b", ylabel=f"Random minus structured {metric} Brier", title=f"RQ2 | d={d}, paired 95% Student intervals")
            ax.grid(alpha=0.2); ax.legend(fontsize=9)
            name=f"rq2_{protocol}_d{d}"; _save(fig, figures, name); made.append(name)
    path = out/"rq3.csv"
    if path.exists():
        rows = read_table(path)
        for eta in sorted({float(r["eta"]) for r in rows}):
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for pool in ("short", "long"):
                vals = sorted((r for r in rows if r["pool"] == pool and float(r["eta"]) == eta), key=lambda r: int(r["budget"]))
                for column, label, linestyle, marker in (("fixed", "fixed", "--", "s"), ("two_batch", "two-batch", "-", "o"), ("adaptive", "adaptive", ":", "x")):
                    ax.plot([int(r["budget"]) for r in vals], [float(r[column+"_unqueried"]) for r in vals],
                            label=f"{pool} check: {label}", linestyle=linestyle, marker=marker, markersize=4)
            ax.set(xlabel="Charged label budget b", ylabel="Exact unqueried Brier risk", title=f"RQ3 | eta={eta:g}; all acquisition classes")
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
            name=f"rq3_eta{eta:g}"; _save(fig, figures, name); made.append(name)
    path=out/"rq4_relations.csv"
    if path.exists():
        rows=read_table(path); fig, ax=plt.subplots(figsize=(7.4,4.6))
        for probe, label in (("length_four", "Length 4, rank 7"), ("length_eight", "Length 8, rank 7"), ("independent", "Independent, rank 8")):
            vals=sorted((r for r in rows if r["probe"]==probe),key=lambda r:float(r["eta"]))
            ax.plot([float(r["eta"]) for r in vals], [float(r["testing_error"]) for r in vals], label=label)
        ax.set(xlabel="Noise rate eta",ylabel="Exact equal-prior testing error",title="RQ4 | Eight-label relation evidence")
        ax.grid(alpha=0.2); ax.legend();_save(fig,figures,"rq4_relation_evidence");made.append("rq4_relation_evidence")
    path=out/"rq4_noise_scale.csv"
    if path.exists():
        rows=read_table(path);fig,ax=plt.subplots(figsize=(7.4,4.6))
        for L in (6,16,64,256):
            vals=sorted((r for r in rows if int(r["L"])==L and float(r["normalized_noise"])>0),key=lambda r:float(r["normalized_noise"]))
            ax.plot([float(r["normalized_noise"]) for r in vals], [float(r["testing_error"]) for r in vals],label=f"log2(r)={L}")
        ax.axvline(1,linestyle=":",linewidth=1)
        ax.set(xscale="log",xlabel="Normalized noise eta / rho_r",ylabel="Exact binomial testing error",title="RQ4 | Analytic parity diagnostic (not a pool simulation)")
        ax.grid(alpha=0.2);ax.legend(fontsize=9);_save(fig,figures,"rq4_noise_scale");made.append("rq4_noise_scale")
    return made
