#!/usr/bin/env python3
"""Aggregate results/jev/jev_*.json into a summary table (CSV + markdown).

Prints a markdown table to stdout and writes results/jev/summary_jev.csv.
Column order matches docs/jev-on-jetson.md.
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPARAMS = {  # rough totals for the on-disk judges
    "openelm-270m": 0.27, "openelm-450m": 0.45, "openelm-1.1b": 1.1,
    "openelm-3b": 3.0, "Qwen3-0.6B-Q8_0": 0.6,
    "qwen35-0.8b": 0.8, "qwen35-2b": 2.0, "qwen35-4b": 4.0,
}


def collect_verify():
    """noul verification runs -> second table + summary_verify.csv."""
    rows = []
    for path in sorted(glob.glob(f"{ROOT}/results/jev/verify_*.json")):
        with open(path) as f:
            s = json.load(f)["summary"]
        rows.append({
            "judge": s["judge"], "target": s["target"],
            "target_acc": s["target_acc"], "n": s["n"],
            "verify_acc": s["verify_acc"], "auc": s["auc"], "ece": s["ece"],
            "p_yes_ok": s["mean_p_yes_correct"], "p_yes_bad": s["mean_p_yes_wrong"],
            "sel_acc_cov75": s["sel_acc_cov75"], "sel_acc_cov50": s["sel_acc_cov50"],
            "seconds": s["seconds"],
        })
    if not rows:
        return
    with open(f"{ROOT}/results/jev/summary_verify.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["judge", "target", "target_acc", "verify_acc", "auc", "ece",
            "sel_acc_cov75", "sel_acc_cov50"]
    print("\n## noul verification (GSM8K candidate answers)\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\n-> {ROOT}/results/jev/summary_verify.csv")


def main():
    rows = []
    for path in sorted(glob.glob(f"{ROOT}/results/jev/jev_*.json")):
        with open(path) as f:
            d = json.load(f)
        s = d["summary"]
        rows.append({
            "model": s["tag"],
            "n_params_b": NPARAMS.get(s["tag"], ""),
            "jev_acc": s["acc"],
            "gen_acc": s.get("gen_arc_acc", ""),
            "delta": round(s["acc"] - s["gen_arc_acc"], 4)
            if s.get("gen_arc_acc") is not None else "",
            "ece": s["ece"], "brier": s["brier"],
            "mean_top1": s["mean_top1"], "coverage": s["mean_coverage"],
            "agreement": s.get("agreement", ""),
            "seconds": s["seconds"], "n": s["n"],
        })
    if not rows:
        print("no results/jev/jev_*.json found")
    else:
        with open(f"{ROOT}/results/jev/summary_jev.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        cols = ["model", "n_params_b", "jev_acc", "gen_acc", "delta", "ece",
                "brier", "agreement", "seconds"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
        print(f"\n-> {ROOT}/results/jev/summary_jev.csv")
    collect_verify()


if __name__ == "__main__":
    main()
