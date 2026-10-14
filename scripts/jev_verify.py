#!/usr/bin/env python3
"""JEV noul task: binary answer verification via next-token probabilities
(LLM-as-Jev, arXiv 2610.02076).

A judge model decides whether a candidate answer is correct, without free-form
generation: the question and the proposed answer go into the user message with
options "[1] yes [2] no", the chat prompt is prefilled with "Best answer: [",
and p(yes) is read from the next-token distribution (one /completion call).

Targets are the GSM8K generations already stored by bench_quality.py
(results/raw/quality_<target>_*.json, latest wins) - so weights of the target
model are NOT needed: one on-disk judge quality-gates any past run, including
the Strata 125B engine.

Metrics: verify accuracy (argmax), AUC of p(yes) as a correctness
discriminator, binary ECE/Brier, and selective accuracy at reduced coverage
(trust the verifier on its top-confidence slice - the edge routing signal).

Usage:
  python3 scripts/jev_verify.py --model work/zoo/gguf/qwen35-4b.gguf \
      --judge-tag qwen35-4b --target hunyuan-4b
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_score import ece_bins, jev_distribution, render_prompt, wait_health

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gsm8k(data_dir, n, seed):
    import pandas as pd
    df = pd.read_parquet(f"{data_dir}/gsm8k/main/test-00000-of-00001.parquet")
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_target(target):
    """Latest quality_<target>_*.json (timestamped) or quality_<target>.json."""
    hits = sorted(glob.glob(f"{ROOT}/results/raw/quality_{target}_*.json"))
    exact = f"{ROOT}/results/raw/quality_{target}.json"
    if os.path.exists(exact):
        hits.append(exact)
    if not hits:
        raise SystemExit(f"no quality file for target '{target}'")
    with open(hits[-1]) as f:
        d = json.load(f)
    return d, os.path.basename(hits[-1])


def verify_one(port, question, pred, n_probs=100):
    q = (f"Question: {question}\n\n"
         f"Proposed answer: {pred}\n\n"
         "Is the proposed answer to the question correct? [1] yes [2] no")
    rendered = render_prompt(port, q)
    probs, coverage = jev_distribution(port, rendered, k=2, n_probs=n_probs)
    return (probs[1] if probs else None), coverage


def auc_score(pairs):
    """Mann-Whitney AUC with average ranks for ties; pairs of (score, label)."""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda t: t[0])
    n, i, rank_sum_pos = len(ranked), 0, 0.0
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank of the tie block
        for _, y in ranked[i:j]:
            if y:
                rank_sum_pos += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def selective_acc(records, coverage):
    rs = sorted(records, key=lambda r: -r["p_yes"])
    take = rs[:max(1, math.ceil(coverage * len(rs)))]
    return sum(r["ok"] for r in take) / len(take)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="judge GGUF path or 'server'")
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--target", required=True, help="quality_<target>_*.json tag")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--data", default=os.path.join(ROOT, "work", "data"))
    ap.add_argument("--n-probs", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    tgt, tgt_file = load_target(args.target)
    gsm = tgt.get("gsm8k") or []
    meta = tgt.get("meta", {})
    n, seed = len(gsm), meta.get("seed", 42)
    questions = load_gsm8k(args.data, n, seed)  # same shuffle; targets may slice

    proc = None
    if args.model != "server":
        proc = subprocess.Popen(
            [os.path.join(ROOT, "work/llama-build/bin/llama-server"),
             "-m", args.model, "--port", str(args.port), "-c", str(args.ctx),
             "-t", str(args.threads), "-ngl", "99", "--no-warmup"],
            stdout=open(f"/tmp/jev-verify-{args.port}.log", "w"),
            stderr=subprocess.STDOUT)
        if not wait_health(args.port):
            proc.kill()
            sys.exit(f"llama-server failed, see /tmp/jev-verify-{args.port}.log")

    try:
        records, t0 = [], time.time()
        for i, row in enumerate(gsm):
            gold_row = questions[row["i"]]
            if row["pred"] is None:
                continue  # unparseable candidate: nothing to verify
            p_yes, coverage = verify_one(args.port, gold_row["question"],
                                         row["pred"], args.n_probs)
            rec = {"i": row["i"], "pred": row["pred"], "gold": row["gold"],
                   "ok": bool(row["ok"]), "coverage": round(coverage, 4)}
            if p_yes is not None:
                rec.update({"p_yes": round(p_yes, 4),
                            "verify_ok": (p_yes > 0.5) == bool(row["ok"])})
            else:
                rec.update({"p_yes": None, "verify_ok": False})
            records.append(rec)
            if (i + 1) % 25 == 0:
                used = [r for r in records if r["p_yes"] is not None]
                acc = sum(r["verify_ok"] for r in used) / max(1, len(used))
                print(f"  {i+1}/{len(gsm)} verify_acc={acc:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        ok = [r for r in records if r["p_yes"] is not None]
        pairs = [(r["p_yes"], r["ok"]) for r in ok]
        ece, _ = ece_bins([{"top1": r["p_yes"], "correct": r["ok"]} for r in ok])
        brier = sum((r["p_yes"] - (1.0 if r["ok"] else 0.0)) ** 2 for r in ok) / len(ok)
        pos = [r["p_yes"] for r in ok if r["ok"]]
        neg = [r["p_yes"] for r in ok if not r["ok"]]
        summary = {
            "judge": args.judge_tag, "target": args.target, "target_file": tgt_file,
            "target_acc": tgt.get("gsm8k_acc"), "n": len(gsm), "n_used": len(ok),
            "verify_acc": round(sum(r["verify_ok"] for r in ok) / len(ok), 4),
            "auc": round(auc_score(pairs), 4) if pairs else None,
            "ece": round(ece, 4), "brier": round(brier, 4),
            "mean_p_yes_correct": round(sum(pos) / len(pos), 4) if pos else None,
            "mean_p_yes_wrong": round(sum(neg) / len(neg), 4) if neg else None,
            "sel_acc_cov100": round(sum(r["ok"] for r in ok) / len(ok), 4),
            "sel_acc_cov75": round(selective_acc(ok, 0.75), 4),
            "sel_acc_cov50": round(selective_acc(ok, 0.5), 4),
            "seconds": round(time.time() - t0, 1),
        }
        outp = os.path.join(ROOT, "results", "jev",
                            f"verify_{args.judge_tag}__{args.target}.json")
        with open(outp, "w") as f:
            json.dump({"meta": vars(args), "summary": summary, "records": records},
                      f, indent=1)
        print(json.dumps(summary, indent=1))
        print(f"-> {outp}")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
