#!/usr/bin/env python3
"""strata-jetson: edge inference benchmark for an OpenAI-compatible local endpoint.

Measures, per prompt length bucket:
  - TTFT (time to first token, streaming, seconds)
  - prefill throughput (prompt tokens / TTFT)
  - decode throughput (completion tokens / generation wall time)
  - peak unified memory & power come from power_sampler.py running alongside.

Usage:
  bench_perf.py --base-url http://127.0.0.1:8080/v1 --model strata \
      --buckets 128,2048,8192 --gen-tokens 256 --repeats 3 --out results/raw/perf_<tag>.json
"""
import argparse, json, statistics, time
from openai import OpenAI

PROMPT_WORD = " Write a short paragraph about the history of computing."


def make_prompt(approx_tokens: int) -> str:
    # ~1.3 word/token for English; repeat to approximate length cheaply and deterministically.
    reps = max(1, approx_tokens * 3 // (4 * len(PROMPT_WORD.split()) + 8))
    return ("You are a helpful assistant. Summarize the following text in one sentence." +
            PROMPT_WORD * reps)


def run_once(cli, model, prompt, gen_tokens):
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    stream = cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=gen_tokens,
        temperature=0,
        stream=True,
    )
    for ev in stream:
        if not ev.choices:
            continue
        d = ev.choices[0].delta
        if getattr(d, "content", None):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_out += 1
        if ev.choices[0].finish_reason:
            break
    total = time.perf_counter() - t0
    return ttft, total, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--buckets", default="128,2048,8192")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default="sk-local")
    args = ap.parse_args()

    cli = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results = {"meta": vars(args), "runs": []}
    for b in [int(x) for x in args.buckets.split(",")]:
        prompt = make_prompt(b)
        n_in = len(prompt) // 4  # approx; exact count read from final usage if provided
        rows = []
        for r in range(args.repeats):
            ttft, total, n_out = run_once(cli, args.model, prompt, args.gen_tokens)
            rows.append({"bucket_tokens": b, "repeat": r, "ttft_s": ttft, "total_s": total,
                         "gen_tokens": n_out, "prompt_tokens_approx": n_in,
                         "prefill_tps": n_in / ttft if ttft else None,
                         "decode_tps": n_out / (total - ttft) if ttft and total > ttft else None})
            print(rows[-1])
        results["runs"].append(rows)
    med = {}
    for rows in results["runs"]:
        b = rows[0]["bucket_tokens"]
        med[b] = {"ttft_p50_s": statistics.median(x["ttft_s"] for x in rows),
                  "prefill_tps_med": statistics.median(x["prefill_tps"] for x in rows),
                  "decode_tps_med": statistics.median(x["decode_tps"] for x in rows)}
    results["summary"] = med
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(med, indent=1))


if __name__ == "__main__":
    main()
