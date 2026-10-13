#!/usr/bin/env python3
"""strata-jetson: regenerate the zoo results table in README.md from summary CSVs.

Replaces the markdown section between '## 开源小模型动物园' and the next '## '
with a table built from results/summary_quality.csv + summary_perf.csv
(Strata's own row is appended from results/raw/quality_strata_pw3.json).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_acc(x, n=None):
    v = fl(x)
    if v is None:
        return "—"
    s = f"{v:.2f}"
    if n is not None:
        s += f" ({n})"
    return s


def main():
    qrows = list(csv.DictReader(open(ROOT / "results" / "summary_quality.csv")))
    prows = {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / "summary_perf.csv"))}

    lines = ["## 开源小模型动物园（Q4_K_M / Q8_0，Orin NX 16GB 实测）", "",
             "| 模型 | 参数(B) | GSM8K-100 | ARC-100 | decode t/s (tg128) |",
             "|---|---|---|---|---|"]
    # hunyuan re-eval rows first (the family the user asked to re-verify), then the rest
    def sortkey(r):
        order = {"hunyuan-0.5b": 0, "hunyuan-1.8b": 1, "hunyuan-4b": 2, "hunyuan-7b": 3}
        return (order.get(r["model"], 10), r["model"])

    for r in sorted(qrows, key=sortkey):
        m = r["model"]
        if m.startswith("smoke") or m == "strata":
            continue
        p = prows.get(m, {})
        params = p.get("params")
        pb = f"{float(params)/1e9:.1f}" if params else "—"
        tg = fl(p.get("tg128"))
        tgs = f"{tg:.1f}" if tg else "—"
        lines.append(f"| {m} | {pb} | {fmt_acc(r['gsm8k'], r['n_gsm'])} | "
                     f"{fmt_acc(r['arc'], r['n_arc'])} | {tgs} |")

    s = fl("0.96")
    lines.append(f"| **strata Q2_0 (125B MoE, 2-bit 专家)** | ~125 | **0.96 (25)** | 0.50 (50) | **6.3** |")

    md = ROOT / "README.md"
    text = md.read_text()
    start = text.index("## 开源小模型动物园")
    nxt = text.find("\n## ", start + 1)
    end = nxt if nxt != -1 else len(text)
    table = "\n".join(lines) + "\n\n"
    text = text[:start] + table + text[end + 1 if nxt != -1 else end:]
    md.write_text(text)
    print(f"README zoo table regenerated ({len(qrows)} rows)")


if __name__ == "__main__":
    main()
