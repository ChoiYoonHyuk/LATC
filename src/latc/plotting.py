"""Plot generated CSVs only; never substitute manuscript numbers for computations.

Each chart is a separate figure. Matplotlib's default color cycle is retained.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finish(fig, ax, path, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_plots(rq: str, out: Path) -> None:
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    if rq == "rq1":
        rows = _read(out / "curves.csv")
        for d in sorted({int(x["d"]) for x in rows}):
            data = sorted([x for x in rows if int(x["d"]) == d], key=lambda x:int(x["b"]))
            fig, ax = plt.subplots(figsize=(7,4.5))
            for key in ("uniform","fixed","subcube","two_batch","adaptive"):
                ax.plot([int(x["b"]) for x in data],[float(x[key]) for x in data],marker="o",markersize=3,label=key)
            _finish(fig,ax,folder/f"cp{d}_curves.png","Acquired labels","Unqueried Brier",f"Noiseless full access: CP-{d}")
    elif rq == "rq2":
        rows = _read(out / "paired_contrasts.csv")
        keys = sorted({(x["study"],int(x["d"]),x["contrast"]) for x in rows if x["metric"]=="conditional_common"})
        for study,d,contrast in keys:
            data = [x for x in rows if (x["study"],int(x["d"]),x["contrast"],x["metric"])==(study,d,contrast,"conditional_common")]
            fig,ax = plt.subplots(figsize=(7,4.5))
            for eta in sorted({float(x["eta"]) for x in data}):
                curve = sorted([x for x in data if float(x["eta"])==eta],key=lambda x:int(x["b"]))
                ax.errorbar([int(x["b"]) for x in curve],[float(x["mean"]) for x in curve],
                            yerr=[float(x["half_width"]) for x in curve],marker="o",capsize=3,label=f"eta={eta:g}")
            _finish(fig,ax,folder/f"{study}_d{d}_{contrast}.png","Acquired labels","Conditional Brier difference",
                    f"{contrast.replace('_',' ')}; d={d}; pointwise 95% paired intervals")
    elif rq == "rq3":
        rows = [x for x in _read(out/"exact_curves.csv") if x["metric"]=="common"]
        for eta in sorted({float(x["eta"]) for x in rows}):
            fig,ax = plt.subplots(figsize=(7,4.5))
            for length in sorted({int(x["check_length"]) for x in rows}):
                data=sorted([x for x in rows if float(x["eta"])==eta and int(x["check_length"])==length],key=lambda x:int(x["b"]))
                for method in ("fixed","two_batch","adaptive"):
                    ax.plot([int(x["b"]) for x in data],[float(x[method]) for x in data],marker="o",markersize=3,label=f"length {length}: {method}")
            _finish(fig,ax,folder/f"matched_eta{eta:g}.png","Acquired labels","Common-target Brier",f"Matched-size/rank pools; eta={eta:g}")
    elif rq == "rq4":
        rows = _read(out/"certificates.csv")
        if rows:
            fig,ax = plt.subplots(figsize=(7,4.5))
            for method in sorted({x["method"] for x in rows}):
                data=sorted([x for x in rows if x["method"]==method],key=lambda x:int(x["r"]))
                for bound in ("random_lower","structured_upper"):
                    ax.plot([int(x["r"]) for x in data],[float(x[bound]) for x in data],marker="o",label=f"{method}: {bound}")
            ax.set_xscale("log",base=2)
            _finish(fig,ax,folder/"finite_risk_bounds.png","Parameter dimension r","Population Brier bound","Finite risk bounds; eta=0.05, p=0.5")
        rows = _read(out/"flat_availability.csv")
        if rows:
            fig,ax=plt.subplots(figsize=(7,4.5))
            ax.plot([int(x["r"]) for x in rows],[float(x["expected_flats"]) for x in rows],marker="o")
            ax.set_yscale("log")
            _finish(fig,ax,folder/"flat_availability.png","Parameter dimension r","Expected contained flats","Eight-point affine flats in q=r^2 random pools")
