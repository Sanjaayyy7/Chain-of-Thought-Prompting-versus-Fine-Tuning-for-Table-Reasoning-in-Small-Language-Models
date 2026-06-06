#!/usr/bin/env python3
"""Export the 6 FINAL_REPORT.ipynb figures as vector PDFs for the LaTeX paper.

Reuses the notebook's exact plotting code so the figures match the report.
Run: .venv/bin/python scripts/export_paper_figures.py
"""
import os
import sys
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402,F401  (kept: notebook cells reference pandas)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "results").is_dir())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import analysis  # noqa: E402
from src.report_fill import _tabfact_decomp, _tabfact_floor  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "axes.axisbelow": True,
                     "grid.alpha": 0.3, "font.size": 10})
RESULTS = analysis.load_results()


def _pick(cond, short, task="wtq"):
    return [r for r in RESULTS if r["condition"] == cond and r["task"] == task
            and r["model"].split("/")[-1] == f"flan-t5-{short}"]


def _ms(vals):
    vals = list(vals)
    return (st.fmean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (None, None)


# --- Figure 1 (notebook cell-08) ---
conds = ["baseline", "cot_plain", "cot_structured", "finetune_answers", "finetune_traces"]
labels = ["Baseline", "CoT plain", "CoT struct", "FT answers", "FT traces"]


def em_of(cond, short):
    m, s = _ms(r["metrics"]["exact_match"] for r in _pick(cond, short))
    return (0.0 if m is None else m), (0.0 if s is None else s)


base = [em_of(c, "base") for c in conds]
large = [em_of(c, "large") for c in conds]
x = np.arange(len(conds))
w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(x - w / 2, [m for m, _ in base], w, yerr=[s for _, s in base], capsize=3, label="flan-t5-base")
ax.bar(x + w / 2, [m for m, _ in large], w, yerr=[s for _, s in large], capsize=3, label="flan-t5-large")
ax.axhline(max(m for m, _ in large), ls="--", c="gray", lw=1)
ax.annotate("best system: large baseline", (0.5, max(m for m, _ in large)),
            textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=12)
ax.set_ylabel("Exact Match")
ax.set_title("WTQ exact match by condition (mean +/- std, 2 seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 2 (cell-10) ---
pairs = [("cot_plain", "base"), ("cot_plain", "large"),
         ("cot_structured", "base"), ("cot_structured", "large")]
plabs = ["plain/base", "plain/large", "struct/base", "struct/large"]
emv = [_ms(r["metrics"]["exact_match"] for r in _pick(c, s))[0] for c, s in pairs]
f1v = [_ms(r["metrics"]["token_f1"] for r in _pick(c, s))[0] for c, s in pairs]
x = np.arange(len(pairs))
w = 0.38
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.bar(x - w / 2, emv, w, label="Exact Match", color="#c66")
ax.bar(x + w / 2, f1v, w, label="Token-F1", color="#69a")
ax.set_xticks(x)
ax.set_xticklabels(plabs)
ax.set_ylabel("score")
ax.set_title("CoT: token-F1 > exact match (the format gap)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 3 (cell-12) ---
floor = _tabfact_floor(RESULTS)
gen = [("generalization_baseline", "Untrained base (floor)"),
       ("generalization_finetune_answers", "FT answers"),
       ("generalization_finetune_traces", "FT traces")]
mods = [lbl for _, lbl in gen]
accs = [_ms(r["metrics"]["classification_accuracy"] for r in RESULTS
            if r["condition"] == c and r["task"] == "tabfact")[0] for c, _ in gen]
covs = [(lambda d: d["coverage_n"] / d["n"])(_tabfact_decomp(RESULTS, c)) for c, _ in gen]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.0))
a1.bar(mods, accs, color="#c44")
a1.axhline(floor, ls="--", c="k", lw=1.2, label=f"floor {floor:.3f}")
a1.set_ylabel("TabFact accuracy")
a1.set_title("Accuracy vs majority floor")
a1.tick_params(axis="x", rotation=12)
a1.legend()
a2.bar(mods, covs, color="#48a")
a2.set_ylabel("fraction mappable to true/false")
a2.set_title("Output-format compliance")
a2.tick_params(axis="x", rotation=12)
fig.tight_layout()
fig.savefig(OUT / "fig3.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 4 (cell-15) ---
from collections import defaultdict  # noqa: E402

agg = defaultdict(lambda: defaultdict(int))
for r in RESULTS:
    if r["task"] != "wtq" or not r["error_distribution"]:
        continue
    key = (r["condition"], r["model"].split("/")[-1])
    for k, v in r["error_distribution"].items():
        agg[key][k] += v
econds = [("baseline", "flan-t5-base"), ("baseline", "flan-t5-large"),
          ("cot_plain", "flan-t5-base"), ("cot_structured", "flan-t5-large"),
          ("finetune_answers", "flan-t5-base"), ("finetune_traces", "flan-t5-base")]
share = {}
for key in econds:
    e = agg[key]
    wrong = sum(e.values()) - e.get("correct", 0)
    share[key] = {t: e.get(t, 0) / wrong for t in ("lookup", "aggregation", "multi_hop")}
xlab = [f"{c[0]} ({c[1].split('-')[-1]})" for c in econds]
x = np.arange(len(econds))
bottom = np.zeros(len(econds))
fig, ax = plt.subplots(figsize=(10.0, 3.2))
for t, c in [("lookup", "#88c"), ("aggregation", "#e88"), ("multi_hop", "#8c8")]:
    vals = [share[k][t] for k in econds]
    ax.bar(x, vals, bottom=bottom, label=t, color=c)
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels(xlab, rotation=20, ha="right")
ax.set_ylabel("share of wrong answers")
ax.set_title("WTQ error-type distribution")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig4.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 5 (cell-17) ---
cq = json.loads((ROOT / "results" / "chain_quality.json").read_text())
fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.bar(["Rater A\n(strict)", "Rater B\n(lenient)"], [cq["mean_a"], cq["mean_b"]], color=["#a55", "#5a8"])
ax.set_ylim(0, 2)
ax.set_ylabel("mean score (0 to 2)")
ax.set_title(f"Chain quality (n={cq['n']}), Cohen kappa = {cq['kappa']:.3f}")
fig.tight_layout()
fig.savefig(OUT / "fig5.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 6 (cell-21) ---
comp = {}
for r in RESULTS:
    key = f"{r['condition']}/{r['model'].split('/')[-1].split('-')[-1]}"
    comp.setdefault(key, []).append(r["compute"]["seconds_per_example"])
items = sorted(((k, st.fmean(v)) for k, v in comp.items()), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(10.0, 4.0))
ax.barh([k for k, _ in items], [v for _, v in items], color="#69a")
ax.set_xlabel("seconds / example")
ax.set_title("Inference cost by condition (mean over seeds)")
fig.tight_layout()
fig.savefig(OUT / "fig6.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote", *(p.name for p in sorted(OUT.glob("fig*.pdf"))))
