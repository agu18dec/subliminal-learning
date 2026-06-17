"""Summarize/plot Phase-2 sweep results.

Reads results/sweep.csv, writes:
  - results/<group>.png  : student_ghost (aux-only) vs the swept value, with the
    cross-model aux-only line for reference and the Phase-1 baseline marked.
  - results/sweep_summary.md : a markdown table per group (paste-able into REPORT).

Usage: python mlp/plot.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
CSV_PATH = os.path.join(RES, "sweep.csv")
BASELINE_AUX = 0.537  # Phase-1 Student (aux. only)


def load() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            groups[row["group"]].append(row)
    return groups


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    groups = load()
    md = ["# Phase-2 sweep summary", "",
          f"Phase-1 baseline Student (aux. only) = **{BASELINE_AUX:.3f}**", ""]

    best_overall = (None, -1.0)
    for group, rows in groups.items():
        md += [f"## {group}", "",
               "| value | student_ghost | ±CI | xmodel_ghost | teacher |",
               "|---|---|---|---|---|"]
        xs, ys, es = [], [], []
        for r in rows:
            sg, ci = _f(r["student_ghost"]), _f(r["student_ghost_ci"])
            md.append(f"| {r['value']} | {sg:.3f} | {ci or 0:.3f} | "
                      f"{_f(r['xmodel_ghost']):.3f} | {_f(r['teacher']):.3f} |")
            xs.append(r["value"]); ys.append(sg); es.append(ci or 0)
            if sg is not None and sg > best_overall[1]:
                best_overall = (f"{group}={r['value']}", sg)
        md.append("")

        # plot
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=4,
                    label="Student (aux. only)")
        ax.axhline(BASELINE_AUX, ls=":", c="gray", label=f"baseline {BASELINE_AUX:.2f}")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(group); ax.set_ylabel("aux-only test acc")
        ax.set_title(f"Aux-only accuracy vs {group}")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(RES, f"{group}.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print("wrote", out)

    md += ["---", f"**Best single config:** {best_overall[0]} "
           f"-> aux-only {best_overall[1]:.3f}"]
    with open(os.path.join(RES, "sweep_summary.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote", os.path.join(RES, "sweep_summary.md"))
    print("BEST:", best_overall)


if __name__ == "__main__":
    main()
