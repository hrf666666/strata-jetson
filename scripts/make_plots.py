#!/usr/bin/env python3
"""strata-jetson: summary charts from results/summary_*.csv + the Strata edge numbers.

Outputs results/plots/*.png:
  zoo_decode_vs_params.png   decode t/s vs parameter count (log-x), labeled
  zoo_pareto.png             GSM8K accuracy vs decode t/s (quality-speed Pareto)
  strata_edge_desktop.png    Strata Q2_0: this Jetson vs the paper's desktop numbers
"""
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def zoo_perf():
    rows = read_csv(ROOT / "results" / "summary_perf.csv")
    out = []
    for r in rows:
        p = r["params"]
        if not p:
            continue
        out.append({"model": r["model"], "params": float(p) / 1e9,
                    "tg128": fl(r.get("tg128")), "pp128": fl(r.get("pp128"))})
    return [r for r in out if r["tg128"]]


def zoo_quality():
    rows = read_csv(ROOT / "results" / "summary_quality.csv")
    out = {}
    for r in rows:
        if r["model"] == "smoke":
            continue
        out[r["model"]] = {"g": fl(r["gsm8k"]), "a": fl(r["arc"])}
    return out


def main():
    perf, qual = zoo_perf(), zoo_quality()

    # 1. decode vs params
    plt.figure(figsize=(7, 4.5))
    for r in perf:
        plt.scatter(r["params"], r["tg128"], s=42)
        plt.annotate(r["model"], (r["params"], r["tg128"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("parameters (B, log)")
    plt.ylabel("decode tok/s (tg128, Q4/Q8 on Orin NX 16GB)")
    plt.title("zoo decode speed vs model size - Jetson Orin NX Super (MAXN, CUDA sm_87)")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "zoo_decode_vs_params.png", dpi=130)
    plt.close()

    # 2. quality vs speed pareto
    plt.figure(figsize=(7, 4.5))
    for r in perf:
        q = qual.get(r["model"])
        if not q or q.get("g") is None:
            continue
        plt.scatter(r["tg128"], q["g"], s=42)
        plt.annotate(r["model"], (r["tg128"], q["g"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    plt.xlabel("decode tok/s")
    plt.ylabel("GSM8K accuracy")
    plt.title("quality-speed trade-off (GSM8K-100, greedy)")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "zoo_pareto.png", dpi=130)
    plt.close()

    # 3. Strata edge vs desktop (paper Table 3, Q2_0 @4K)
    labels = ["output tok/s", "prompt tok/s"]
    desktop = [94.6, 539]
    edge_out, edge_pp = 6.31, 38.9
    edge = [edge_out, edge_pp]
    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar([i - .18 for i in x], desktop, width=.36, label="RTX 5070 12GB (paper)")
    plt.bar([i + .18 for i in x], edge, width=.36, label="Orin NX 16GB (ours)", color="tab:green")
    plt.yscale("log")
    plt.xticks(list(x), labels)
    plt.ylabel("tok/s (log)")
    plt.title("Strata Q2_0 (125B): gaming PC vs Jetson edge")
    for i, v in enumerate(desktop):
        plt.text(i - .18, v * 1.1, f"{v:g}", ha="center", fontsize=8)
    for i, v in enumerate(edge):
        plt.text(i + .18, v * 1.1, f"{v:g}", ha="center", fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS / "strata_edge_desktop.png", dpi=130)
    plt.close()
    print(f"plots -> {PLOTS}")


if __name__ == "__main__":
    main()
