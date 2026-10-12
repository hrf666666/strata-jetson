#!/usr/bin/env python3
"""strata-jetson: quality eval (GSM8K-100 + ARC-Challenge-100) against an OpenAI-compatible endpoint.

Fixed sampling (seed 42), greedy decoding. Scores:
  - GSM8K: 8-shot CoT from train split, exact match on the final "#### N" number.
  - ARC-Challenge: 0-shot A/B/C/D letter choice.

Usage:
  bench_quality.py --base-url http://127.0.0.1:8080/v1 --model strata \
      --data work/data --n 100 --out results/raw/quality_<tag>.json
"""
import argparse, json, os, random, re, time
from openai import OpenAI

GSM8K_SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Weng earns 12/60 = <<12/60=0.2>>$0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = <<0.2*50=10>>$10.\n#### 10"),
    ("A deep-sea monster rises from the waters once every hundred years to feast on a ship between 30 and 40 ships. How many years will it take to feast on 250 ships?",
     "The monster rises once every 100 years to feast on between 30 and 40 ships, an average of 35.\n250 / 35 = 7.14, so it will take about 7*100 = 700 years.\n#### 700"),
]


def gsm8k_fewshot():
    parts = []
    for q, a in GSM8K_SHOTS:
        parts.append(f"Question: {q}\nAnswer: {a}")
    return "\n\n".join(parts) + "\n\n"


def gsm8k_answer(text):
    m = re.findall(r"####\s*\$?(-?[0-9,]+(?:\.[0-9]+)?)", text)
    if not m:
        m = re.findall(r"(-?\$?[0-9,]+(?:\.[0-9]+)?)", text.splitlines()[-1]) if text.splitlines() else []
    if not m:
        return None
    v = m[-1].replace(",", "").replace("$", "")
    try:
        return float(v)
    except ValueError:
        return None


def load_gsm8k(data_dir, n, seed):
    import pandas as pd
    df = pd.read_parquet(f"{data_dir}/gsm8k/main/test-00000-of-00001.parquet")
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_arc(data_dir, n, seed):
    import pandas as pd
    df = pd.read_parquet(f"{data_dir}/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet")
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def ask(cli, model, prompt, max_tokens, timeout=600):
    # chat endpoint only: Strata's server does not implement raw /v1/completions
    t0 = time.time()
    out = cli.chat.completions.create(model=model,
                                      messages=[{"role": "user", "content": prompt}],
                                      max_tokens=max_tokens, temperature=0, timeout=timeout)
    m = out.choices[0].message
    text = (getattr(m, "content", None) or "") + "\n" + (getattr(m, "reasoning_content", None) or "")
    return text, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--data", default="work/data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens-gsm", type=int, default=512)
    ap.add_argument("--max-tokens-arc", type=int, default=512)
    ap.add_argument("--only", choices=["gsm", "arc"], default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default="sk-local")
    args = ap.parse_args()
    cli = OpenAI(base_url=args.base_url, api_key=args.api_key)

    res = {"meta": vars(args), "gsm8k": [], "arc": []}
    outp = os.path.abspath(args.out)
    if args.only and os.path.exists(outp):   # keep the other part's existing numbers
        try:
            prev = json.load(open(outp))
            res["gsm8k"] = prev.get("gsm8k", [])
            res["arc"] = prev.get("arc", [])
        except Exception:
            pass

    # ---- GSM8K ----
    shots = gsm8k_fewshot()
    correct = 0
    for i, row in enumerate(load_gsm8k(args.data, args.n, args.seed) if args.only != "arc" else []):
        q = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            text, _ = ask(cli, args.model, f"{shots}Question: {q}\nAnswer:", args.max_tokens_gsm)
        except Exception as e:
            text = f"<error {e}>"
        pred = gsm8k_answer(text)
        ok = pred is not None and abs(pred - float(gold)) < 1e-4
        correct += ok
        res["gsm8k"].append({"i": i, "ok": bool(ok), "pred": pred, "gold": gold, "text": text[:300]})
        if (i + 1) % 20 == 0:
            print(f"gsm8k {i+1}: acc={correct}/{i+1}")
    res["gsm8k_acc"] = (sum(x["ok"] for x in res["gsm8k"]) / len(res["gsm8k"])) if res["gsm8k"] else None

    # ---- ARC-Challenge ----
    correct = 0
    letters = ["A", "B", "C", "D", "E"]
    for i, row in enumerate(load_arc(args.data, args.n, args.seed) if args.only != "gsm" else []):
        ch = row["choices"]
        labels = list(ch["label"])
        texts = list(ch["text"])
        key = row["answerKey"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        prompt = (f"The following is a multiple choice question. Answer with just the letter.\n\n"
                  f"Question: {row['question']}\n{opts}\nAnswer:")
        try:
            text, _ = ask(cli, args.model, prompt, args.max_tokens_arc)
        except Exception as e:
            text = f"<error {e}>"
        # the answer is the LAST standalone letter (thinking text is full of A-E words)
        cands = re.findall(r"\b([A-E])\b", text.strip().upper())
        pred = cands[-1] if cands else "?"
        ok = pred == key
        correct += ok
        res["arc"].append({"i": i, "ok": bool(ok), "pred": pred, "gold": key, "text": text[:200]})
        if (i + 1) % 20 == 0:
            print(f"arc {i+1}: acc={correct}/{i+1}")
    res["arc_acc"] = (sum(x["ok"] for x in res["arc"]) / len(res["arc"])) if res["arc"] else None

    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"gsm8k_acc": res["gsm8k_acc"], "arc_acc": res["arc_acc"]}, indent=1))


if __name__ == "__main__":
    main()
