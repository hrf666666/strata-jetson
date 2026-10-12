#!/usr/bin/env python3
"""strata-jetson: tegrastats sampler -> CSV (RAM/GR3D/power) aligned with benchmark runs.

Usage: power_sampler.py --out results/raw/power_<tag>.csv --interval 200
Stop with Ctrl-C / SIGTERM. Columns: ts, ram_used_mb, swap_mb, gr3d_pct, cpu_pct, power_mw, temp_cpu.
"""
import argparse, csv, datetime, re, signal, subprocess, sys

FIELDS = ("ts ram_used_mb swap_mb gr3d_pct cpu_pct power_mw temp_cpu").split()
RAM = re.compile(r"RAM (\d+)/(\d+)MB")
SWAP = re.compile(r"SWAP (\d+)/")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
CPU = re.compile(r"CPU \[(.*?)\]")
PWR = re.compile(r"VDD_IN (\d+)/(\d+)mW|POM_5V_IN (\d+)/")
TCPU = re.compile(r"cpu@(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=200)
    args = ap.parse_args()
    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(FIELDS)
    p = subprocess.Popen(["tegrastats", "--interval", str(args.interval)], stdout=subprocess.PIPE, text=True)
    signal.signal(signal.SIGTERM, lambda *_: p.kill())

    def cpu_avg(s):
        if not s:
            return ""
        vals = [int(x.split("%")[0]) for x in s.split(",") if "%" in x]
        return round(sum(vals) / len(vals)) if vals else ""

    try:
        for line in p.stdout:
            ram = RAM.search(line)
            if not ram:
                continue
            gr3d = GR3D.search(line)
            cpu = CPU.search(line)
            pwr = re.search(r"VDD_IN (\d+)mW", line)
            tcpu = TCPU.search(line)
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                        ram.group(1), SWAP.search(line).group(1) if SWAP.search(line) else "",
                        gr3d.group(1) if gr3d else "", cpu_avg(cpu.group(1) if cpu else ""),
                        pwr.group(1) if pwr else "", tcpu.group(1) if tcpu else ""])
            f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        p.kill()
        f.close()
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
