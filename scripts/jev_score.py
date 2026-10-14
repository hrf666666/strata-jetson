#!/usr/bin/env python3
"""JEV-style calibrated decision scoring (LLM-as-Jev, arXiv 2610.02076).

Instead of free-form generation + answer parsing, options are numbered [1]..[K],
the chat prompt is prefilled with "Best answer: [", and the decision
distribution is read directly from the next-token probabilities over the
candidate digits. K <= 9 digits differ in the first token, so ONE
llama-server /completion call with n_probs captures the full candidate
distribution (server softmaxes over the whole vocab, so probs are exact).

Metrics: accuracy (argmax), ECE (10 equal-width bins), multiclass Brier,
mean NLL, entropy, candidate coverage. --compare-gen additionally aligns with
the generation-based ARC answers stored by bench_quality.py (same seed/n) and
reports agreement.

Usage:
  python3 scripts/jev_score.py --model work/zoo/gguf/qwen35-4b.gguf --tag qwen35-4b
"""
import argparse
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_arc(data_dir, n, seed):
    import pandas as pd
    df = pd.read_parquet(f"{data_dir}/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet")
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def http_json(url, payload, timeout=600, retries=3):
    body = json.dumps(payload).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


def render_prompt(port, question):
    """Server-side chat-template rendering with thinking disabled; base models
    without a chat template fall back to the raw question text."""
    try:
        r = http_json(f"http://127.0.0.1:{port}/apply-template",
                      {"messages": [{"role": "user", "content": question}],
                       "chat_template_kwargs": {"enable_thinking": False},
                       "add_generation_prompt": True}, retries=1)
        return r["prompt"]
    except Exception:
        return question


def build_question(row):
    """Jev-style user message: options numbered [1]..[K] by position."""
    texts = list(row["choices"]["text"])
    lines = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))
    q = f"{row['question']}\n{lines}"
    gold = list(row["choices"]["label"]).index(row["answerKey"]) + 1  # position, 1-based
    return q, gold, len(texts)


def jev_distribution(port, rendered, k, n_probs=100):
    """One completion call -> {digit: prob} normalized over the k candidates."""
    prompt = rendered + "Best answer: ["
    r = http_json(f"http://127.0.0.1:{port}/completion",
                  {"prompt": prompt, "max_tokens": 1, "n_probs": n_probs,
                   "temperature": 0, "cache_prompt": True})
    entry = r["completion_probabilities"][0]
    cand = {j + 1: 0.0 for j in range(k)}
    coverage = 0.0
    for alt in entry["top_logprobs"]:
        raw = bytes(alt.get("bytes") or []).decode("utf-8", "replace").lstrip()
        if raw and raw[0] in "123456789" and 1 <= int(raw[0]) <= k:
            p = math.exp(alt["logprob"])
            cand[int(raw[0])] += p
            coverage += p
    total = sum(cand.values())
    if total <= 0:
        return None, coverage
    return {d: p / total for d, p in cand.items()}, coverage


def ece_bins(records, n_bins=10):
    """Equal-width confidence bins; returns (ece, bins list)."""
    bins = [[] for _ in range(n_bins)]
    for r in records:
        b = min(n_bins - 1, int(r["top1"] * n_bins))
        bins[b].append(r)
    ece, out = 0.0, []
    for b, items in enumerate(bins):
        if not items:
            continue
        conf = sum(x["top1"] for x in items) / len(items)
        acc = sum(1.0 for x in items if x["correct"]) / len(items)
        ece += len(items) / len(records) * abs(conf - acc)
        out.append({"conf": round(conf, 4), "acc": round(acc, 4), "n": len(items)})
    return ece, out


def brier_score(records):
    # ARC mixes 3/4/5-choice questions: use each record's own k
    return sum(sum((r["probs"][str(d)] - (1.0 if r["gold"] == d else 0.0)) ** 2
                   for d in range(1, r["k"] + 1)) for r in records) / len(records)


def entropy(probs):
    return -sum(p * math.log(p) for p in probs.values() if p > 1e-12)


def load_gen_predictions(model_tag, n, seed):
    """Latest quality_<tag>_*.json with a full arc list -> {i: position_pred}."""
    hits = sorted(glob.glob(f"{ROOT}/results/raw/quality_{model_tag}_*.json"))
    for path in reversed(hits):
        with open(path) as f:
            res = json.load(f)
        if not isinstance(res, dict):
            continue  # llama-bench perf output, not a quality file
        arc = res.get("arc") or []
        # pred letters are original ARC labels; the eval rows were stored in the
        # same order as load_arc(seed, n) -> position via per-row label list
        rows = {r["i"]: r for r in arc}
        if len(rows) < n:
            continue
        import pandas as pd
        df = pd.read_parquet(f"{ROOT}/work/data/ai2_arc/ARC-Challenge/"
                             f"test-00000-of-00001.parquet")
        all_rows = df.to_dict("records")
        random.Random(seed).shuffle(all_rows)
        out = {}
        for i, r in rows.items():
            labels = list(all_rows[i]["choices"]["label"])
            out[i] = labels.index(r["pred"]) + 1 if r["pred"] in labels else None
        return out, res.get("arc_acc")
    return None, None


def wait_health(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="GGUF path (or 'server' if already up)")
    ap.add_argument("--tag", default=None, help="result tag; defaults to model stem")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--data", default=os.path.join(ROOT, "work", "data"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-probs", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--compare-gen", action="store_true",
                    help="align with bench_quality.py generation answers (same seed/n)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
    args = ap.parse_args()
    tag = args.tag or re.sub(r"\.gguf$", "", os.path.basename(args.model))

    proc = None
    if args.model != "server":
        proc = subprocess.Popen(
            [os.path.join(ROOT, "work/llama-build/bin/llama-server"),
             "-m", args.model, "--port", str(args.port), "-c", str(args.ctx),
             "-t", str(args.threads), "-ngl", "99", "--no-warmup"],
            stdout=open(f"/tmp/jev-server-{args.port}.log", "w"),
            stderr=subprocess.STDOUT)
        if not wait_health(args.port):
            proc.kill()
            sys.exit(f"llama-server failed to come up, see /tmp/jev-server-{args.port}.log")

    try:
        rows = load_arc(args.data, args.n, args.seed)[: args.limit or None]
        records = []
        t0 = time.time()
        for i, row in enumerate(rows):
            q, gold, k = build_question(row)
            rendered = render_prompt(args.port, q)
            probs, coverage = jev_distribution(args.port, rendered, k, args.n_probs)
            rec = {"i": i, "gold": gold, "k": k, "coverage": round(coverage, 4)}
            if probs is not None:
                pred = max(probs, key=probs.get)
                rec.update({"pred": pred, "correct": pred == gold,
                            "top1": round(probs[pred], 4),
                            "probs": {str(d): round(p, 5) for d, p in probs.items()},
                            "entropy": round(entropy(probs), 4)})
            else:
                rec.update({"pred": None, "correct": False, "top1": 0.0,
                            "probs": None, "entropy": None})
            records.append(rec)
            if (i + 1) % 25 == 0:
                acc = sum(r["correct"] for r in records) / (i + 1)
                print(f"  {i+1}/{len(rows)} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

        ok = [r for r in records if r["probs"] is not None]
        ece, bins = ece_bins(ok)
        summary = {
            "tag": tag, "task": "arc", "n": len(records), "seed": args.seed,
            "acc": round(sum(r["correct"] for r in records) / len(records), 4),
            "ece": round(ece, 4), "brier": round(brier_score(ok), 4) if ok else None,
            "mean_top1": round(sum(r["top1"] for r in ok) / len(ok), 4) if ok else None,
            "mean_entropy": round(sum(r["entropy"] for r in ok) / len(ok), 4) if ok else None,
            "mean_coverage": round(sum(r["coverage"] for r in records) / len(records), 4),
            "n_failed": len(records) - len(ok),
            "seconds": round(time.time() - t0, 1), "bins": bins,
        }
        if args.compare_gen:
            gen, gen_acc = load_gen_predictions(tag, args.n, args.seed)
            if gen:
                pairs = [(r, gen.get(r["i"])) for r in records if gen.get(r["i"]) is not None]
                agree = sum(r["pred"] == g for r, g in pairs)
                summary["gen_arc_acc"] = gen_acc
                summary["gen_n"] = len(pairs)
                summary["agreement"] = round(agree / len(pairs), 4) if pairs else None
                summary["agree_correct_only"] = round(
                    sum(1 for r, g in pairs if g == r["pred"] == r["gold"]) /
                    max(1, sum(1 for r, g in pairs if r["gold"] == g)), 4)

        outp = os.path.join(ROOT, "results", "jev", f"jev_{tag}.json")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
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
