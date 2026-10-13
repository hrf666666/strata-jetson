#!/usr/bin/env python3
"""strata-jetson: consolidate results/raw/{bench,quality}_*.json into summary CSVs.

Outputs:
  results/summary_perf.csv    one row per model: params, size, pp128/pp2048, tg128, TTFT
  results/summary_quality.csv one row per model: gsm8k_acc, arc_acc
Prints a combined markdown table (perf from llama-bench JSON, quality, and the
Strata row merged from the Strata-specific results).
"""
import csv, glob, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"


def latest(pattern):
    fs = sorted(glob.glob(str(RAW / pattern)))
    return fs[-1] if fs else None


def bench_rows(path):
    """llama-bench -o json: list of dicts with model_params/size_avg/test fields."""
    try:
        d = json.load(open(path))
    except Exception:
        return None, []
    rows, size, params = [], None, None
    for r in d if isinstance(d, list) else []:
        size = size or r.get("model_size")
        params = params or r.get("model_n_params")
        # this llama-bench -o json has no "test" label: derive it from n_prompt/n_gen
        np_, ng_ = r.get("n_prompt"), r.get("n_gen")
        test = r.get("test")
        if not isinstance(test, str):
            test = (f"tg{ng_}" if (ng_ or 0) > 0 else (f"pp{np_}" if np_ else None)) if np_ is not None else None
        rows.append({"test": test, "avg_ts": r.get("avg_ts")})
    return {"size": size, "params": params}, rows


def main():
    perf, qual = [], []
    perf_map = {}   # keep the latest bench run per model
    for bp in sorted(glob.glob(str(RAW / "bench_*.json"))):
        name = os.path.basename(bp)[len("bench_"):].rsplit("_", 1)[0]
        name = name[:-5] if name.endswith(".json") else name   # timestamp-less smoke files
        if name.startswith("Qwen3-0.6B"):
            name = "smoke-0.6b"
        meta, rows = bench_rows(bp)
        if not rows:
            continue
        row = {"model": name, "size_gb": meta["size"], "params": meta["params"]}
        for r in rows:
            # this llama-bench -o json has no "test" label: derive it from n_prompt/n_gen
            test = r.get("test")
            if not isinstance(test, str):
                np_, ng_ = r.get("n_prompt"), r.get("n_gen")
                test = (f"tg{np_}" if (ng_ or 0) > 0 else f"pp{np_}") if np_ else None
            if isinstance(test, str):
                row[test] = round(r["avg_ts"], 1) if r.get("avg_ts") else None
        perf_map[name] = row
    perf = list(perf_map.values())
    qmap = {}   # same model measured twice (e.g. hunyuan before/after a parser fix): keep the LATEST run
    for qp in sorted(glob.glob(str(RAW / "quality_*.json"))):
        name = os.path.basename(qp)[len("quality_"):].rsplit("_", 1)[0]
        name = name[:-5] if name.endswith(".json") else name
        try:
            d = json.load(open(qp))
        except Exception:
            continue
        qmap[name] = {"model": name, "gsm8k": d.get("gsm8k_acc"), "arc": d.get("arc_acc"),
                      "n_gsm": len(d.get("gsm8k") or []), "n_arc": len(d.get("arc") or [])}
    qual = list(qmap.values())

    with open(OUT / "summary_perf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "size_gb", "params", "pp128", "pp2048", "pp8192", "tg128", "tg8192"])
        w.writeheader()
        for r in sorted(perf, key=lambda x: str(x["model"])):
            w.writerow(r)
    with open(OUT / "summary_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "gsm8k", "arc", "n_gsm", "n_arc"])
        w.writeheader()
        for r in sorted(qual, key=lambda x: str(x["model"])):
            w.writerow(r)
    print(f"wrote summary_perf.csv ({len(perf)} rows), summary_quality.csv ({len(qual)} rows)")


if __name__ == "__main__":
    main()
