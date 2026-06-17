"""Phase-2 ablation driver for the MLP/MNIST subliminal-learning experiment.

Runs one-knob-at-a-time sweeps from a baseline Config and records, for each run,
the per-variant mean accuracy (+95% CI). The primary metric is `student_ghost`
("Student (aux. only)") — the pure subliminal channel we want to maximize.

Results are appended to results/sweep.csv (resumable: rows already present for a
(group, value) pair are skipped). Run a subset with --groups.

Examples:
  python mlp/sweep.py --groups m_ghost                 # just the ghost-count sweep
  python mlp/sweep.py --groups m_ghost epochs temp     # several groups
  python mlp/sweep.py --all                            # everything
  python mlp/sweep.py --quick --all                    # fast/cheap smoke version
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, run_experiment, ci_95  # noqa: E402

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "sweep.csv")
FIELDS = [
    "group", "value",
    "student_ghost", "student_ghost_ci",
    "student_all", "student_all_ci",
    "xmodel_ghost", "xmodel_ghost_ci",
    "teacher", "reference",
]

# group -> (config-field, [values to sweep])
SWEEPS: dict[str, tuple[str, list]] = {
    "m_ghost":     ("m_ghost", [1, 2, 3, 5, 10, 20, 50]),
    "epochs":      ("epochs_distill", [5, 10, 20, 50]),
    "width":       ("hidden", [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]),
    "depth":       ("hidden", [(256,), (256, 256), (256, 256, 256), (256, 256, 256, 256)]),
    "temp":        ("temperature", [1.0, 2.0, 4.0, 8.0]),
    "noise":       ("noise_type", ["uniform", "gaussian", "real"]),
    "n_noise":     ("n_noise", [5000, 20000, 60000]),
    "teacher":     ("epochs_teacher", [1, 5, 20]),
    "lr":          ("lr", [1e-4, 3e-4, 1e-3]),
    # Phase-2b init-distance control (probes the "shared init is the carrier" claim)
    "init_alpha":  ("init_alpha", [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]),
}


def load_done() -> set[tuple[str, str]]:
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add((row["group"], row["value"]))
    return done


def append_row(row: dict):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(group: str, field: str, value, base: Config) -> dict:
    cfg = replace(base, **{field: value})
    res = run_experiment(cfg)

    def ms(k):
        return float(np.mean(res[k])) if k in res else None, ci_95(res[k]) if k in res else None

    sg, sg_ci = ms("student_ghost")
    sa, sa_ci = ms("student_all")
    xg, xg_ci = ms("xmodel_ghost")
    return {
        "group": group, "value": str(value),
        "student_ghost": sg, "student_ghost_ci": sg_ci,
        "student_all": sa, "student_all_ci": sa_ci,
        "xmodel_ghost": xg, "xmodel_ghost_ci": xg_ci,
        "teacher": float(np.mean(res["teacher"])),
        "reference": float(np.mean(res["reference"])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=[], choices=list(SWEEPS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true", help="small/cheap: 16 models, 5k noise")
    p.add_argument("--force", action="store_true", help="rerun even if present")
    args = p.parse_args()

    groups = list(SWEEPS) if args.all else args.groups
    if not groups:
        p.error("specify --groups <...> or --all")

    base = Config()
    if args.quick:
        base = replace(base, n_models=16, n_noise=5000)

    done = set() if args.force else load_done()
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    for group in groups:
        field, values = SWEEPS[group]
        for value in values:
            key = (group, str(value))
            if key in done:
                print(f"skip {key} (done)")
                continue
            print(f"running {group}={value} ...", flush=True)
            row = run_one(group, field, value, base)
            append_row(row)
            print(f"  -> student_ghost={row['student_ghost']:.3f} "
                  f"xmodel_ghost={row['xmodel_ghost']:.3f} "
                  f"teacher={row['teacher']:.3f}", flush=True)


if __name__ == "__main__":
    main()
