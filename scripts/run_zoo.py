#!/usr/bin/env python3
"""strata-jetson: model-zoo benchmark runner for Jetson Orin NX.

Per model: obtain GGUF (Q4_K_M unless noted) -> llama-bench on GPU -> llama-server +
GSM8K/ARC eval -> record -> (optionally) delete weights.  Results land in results/raw/.

Model sources, in order: modelscope.cn (fast, domestic) -> hf-mirror.com (slower).
Some models only exist as safetensors on the mirrors: those are converted with
llama.cpp's convert_hf_to_gguf.py then quantized to Q4_K_M with llama-quantize.

Usage:
  run_zoo.py list
  run_zoo.py fetch <name>          # get the GGUF in place
  run_zoo.py bench <name>          # llama-bench (pp/tg at several context buckets)
  run_zoo.py eval  <name>          # llama-server + GSM8K-100 + ARC-100
  run_zoo.py all  <name> [--keep]  # fetch + bench + eval (+ keep weights)
"""
import argparse, json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
GGUF_DIR = WORK / "zoo" / "gguf"
RAW = ROOT / "results" / "raw"
LLAMA = WORK / "llama-build" / "bin"
RESULTS = ROOT / "results" / "raw"

# name, ms_repo, ms_file, hf_repo, hf_file, hf_safetensors_repo (convert path), notes
ZOO = {
    "qwen35-0.8b":  dict(ms=("Qwen/Qwen3.5-0.8B-GGUF", None), hf=("Qwen/Qwen3.5-0.8B-GGUF", None),
                         safetensors="Qwen/Qwen3.5-0.8B", note="new arch; convert if no GGUF found"),
    "qwen35-2b":    dict(ms=("Qwen/Qwen3.5-2B-GGUF", None), hf=("Qwen/Qwen3.5-2B-GGUF", None),
                         safetensors="Qwen/Qwen3.5-2B", note=""),
    "qwen35-4b":    dict(ms=("Qwen/Qwen3.5-4B-GGUF", None), hf=("Qwen/Qwen3.5-4B-GGUF", None),
                         safetensors="Qwen/Qwen3.5-4B", note=""),
    "qwen35-9b":    dict(ms=("Qwen/Qwen3.5-9B-GGUF", None), hf=("Qwen/Qwen3.5-9B-GGUF", None),
                         safetensors="Qwen/Qwen3.5-9B", note="bf16 over RAM; Q4_K_M only"),
    "hunyuan-0.5b": dict(ms=("tencent/Hunyuan-0.5B-Instruct-GGUF", None), hf=("bartowski/tencent_Hunyuan-0.5B-Instruct-GGUF", None),
                         safetensors="tencent/Hunyuan-0.5B-Instruct", note=""),
    "hunyuan-1.8b": dict(ms=("tencent/Hunyuan-1.8B-Instruct-GGUF", None), hf=("bartowski/tencent_Hunyuan-1.8B-Instruct-GGUF", None),
                         safetensors="tencent/Hunyuan-1.8B-Instruct", note=""),
    "hunyuan-4b":   dict(ms=("tencent/Hunyuan-4B-Instruct-GGUF", None), hf=("bartowski/tencent_Hunyuan-4B-Instruct-GGUF", None),
                         safetensors="tencent/Hunyuan-4B-Instruct", note=""),
    "hunyuan-7b":   dict(ms=("tencent/Hunyuan-7B-Instruct-GGUF", None), hf=("bartowski/tencent_Hunyuan-7B-Instruct-GGUF", None),
                         safetensors="tencent/Hunyuan-7B-Instruct", note=""),
    "glm-z1-9b":    dict(ms=("zai-org/GLM-Z1-9B-0414-GGUF", None), hf=("THUDM/GLM-Z1-9B-0414-GGUF", None),
                         safetensors="zai-org/GLM-Z1-9B-0414", note="thinking model; cap eval tokens"),
    "r1-distill-1.5b": dict(ms=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", None), hf=("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", None),
                         safetensors="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", note="thinking"),
    "r1-distill-7b": dict(ms=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF", None), hf=("unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", None),
                         safetensors="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", note="thinking"),
    "openelm-270m": dict(ms=("apple/OpenELM-270M-Instruct-GGUF", None), hf=("apple/OpenELM-270M-Instruct-GGUF", None),
                         safetensors="apple/OpenELM-270M-Instruct", note="convert path likely"),
    "openelm-450m": dict(ms=("apple/OpenELM-450M-Instruct-GGUF", None), hf=("apple/OpenELM-450M-Instruct-GGUF", None),
                         safetensors="apple/OpenELM-450M-Instruct", note=""),
    "openelm-1.1b": dict(ms=("apple/OpenELM-1_1B-Instruct-GGUF", None), hf=("apple/OpenELM-1_1B-Instruct-GGUF", None),
                         safetensors="apple/OpenELM-1_1B-Instruct", note=""),
    "openelm-3b":   dict(ms=("apple/OpenELM-3B-Instruct-GGUF", None), hf=("apple/OpenELM-3B-Instruct-GGUF", None),
                         safetensors="apple/OpenELM-3B-Instruct", note=""),
    "internlm25-1.8b": dict(ms=("internlm/internlm2_5-1_8b-chat-gguf", "internlm2_5-1_8b-chat-q4_k_m.gguf"),
                         hf=("internlm/internlm2_5-1_8b-chat-gguf", "internlm2_5-1_8b-chat-q4_k_m.gguf"),
                         safetensors="internlm/internlm2_5-1_8b-chat", note=""),
    "intern-s1-mini": dict(ms=("internlm/Intern-S1-mini-GGUF", None), hf=("internlm/Intern-S1-mini-GGUF", None),
                         safetensors=None, note="VLM; llama.cpp support to verify"),
    "qwen3vl-2b-ins": dict(ms=("Qwen/Qwen3-VL-2B-Instruct-GGUF", None), hf=("Qwen/Qwen3-VL-2B-Instruct-GGUF", None),
                         safetensors="/home/nvidia/code/Qwen/Qwen3-VL-2B-Instruct", note="local safetensors"),
    "qwen3vl-2b-think": dict(ms=("Qwen/Qwen3-VL-2B-Thinking-GGUF", None), hf=("Qwen/Qwen3-VL-2B-Thinking-GGUF", None),
                         safetensors="/home/nvidia/code/Qwen/Qwen3-VL-2B-Thinking", note="local safetensors"),
}

MS_API = "https://modelscope.cn/api/v1/models/{repo}/repo?FilePath={path}"
MS_LIST = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
HF_RESOLVE = "https://hf-mirror.com/{repo}/resolve/main/{path}"


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), **kw)


def _curl_json(url, tries=5):
    # the mirror's TLS drops connections sporadically: retry through curl, which rode out the flakiness
    # during the big download far better than one-shot urllib.
    import subprocess, time as _t
    for i in range(tries):
        r = subprocess.run(["curl", "-s", "-m", "25", url], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                pass
        _t.sleep(4 * (i + 1))
    raise RuntimeError(f"curl list failed after {tries}: {url[:80]}")


def ms_files(repo):
    d = _curl_json(MS_LIST.format(repo=repo))
    if d.get("Success") is False or "Files" not in d.get("Data", {}):
        raise RuntimeError(d.get("Message", "no Files"))
    return [f["Path"] for f in d["Data"]["Files"]]


def hf_files(repo):
    d = _curl_json(f"https://hf-mirror.com/api/models/{repo}")
    return [s["rfilename"] for s in d.get("siblings", [])]


def find_gguf(repo_list, want="Q4_K_M"):
    cands = [f for f in repo_list if f.lower().endswith(".gguf")]
    q4 = [f for f in cands if "q4_k_m" in f.lower()]
    q4s = [f for f in cands if "q4_k_s" in f.lower()]
    q8 = [f for f in cands if "q8_0" in f.lower()]
    pick = (q4 or q4s or q8 or [c for c in cands if "IQ4" in c]) if want == "Q4_K_M" else cands
    return pick[0] if pick else None


def fetch(name):
    m = ZOO[name]
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    local = GGUF_DIR / f"{name}.gguf"
    if local.exists() and local.stat().st_size > 1e6:
        print(f"[{name}] already here: {local}")
        return local
    for src, api in (("ms", m["ms"]), ("hf", m["hf"])):
        if not src or not api or api[0] is None:
            continue
        repo, fixed = api
        try:
            files = [fixed] if fixed else None
            if files is None:
                files = ms_files(repo) if src == "ms" else hf_files(repo)
        except Exception as e:
            print(f"[{name}] list failed on {src}: {e}")
            continue
        f = find_gguf(files)
        if not f:
            continue
        url = (MS_API.format(repo=repo, path=f.replace("/", "%2F").replace("?", "%3F"))
               if src == "ms" else HF_RESOLVE.format(repo=repo, path=f))
        print(f"[{name}] {src}: {repo}/{f}")
        r = sh(["aria2c", "-x8", "-s8", "-k1M", "--file-allocation=none",
                "--console-log-level=warn", "--summary-interval=0",
                "--max-tries=0", "--retry-wait=10", "--dir", GGUF_DIR, "-o", local.name, url])
        if r.returncode == 0:
            return local
    # fallback: convert from safetensors
    if m.get("safetensors"):
        return convert(name, m["safetensors"])
    print(f"[{name}] no GGUF source found")
    return None


def convert(name, repo_or_path):
    out = GGUF_DIR / f"{name}-f16.gguf"
    src = repo_or_path
    if repo_or_path.startswith("/"):
        src = Path(repo_or_path)
        if not src.exists():
            print(f"[{name}] local path missing: {src}")
            return None
    else:
        src = WORK / "zoo" / "hf" / name
        src.mkdir(parents=True, exist_ok=True)
        if not list(src.glob("*.safetensors")):   # already downloaded: skip the network entirely
            py = os.environ.get("PY310", str(Path.home() / "miniconda3/envs/py310/bin"))
            env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com")
            sh([py + "/hf", "download", repo_or_path, "--local-dir", src], env=env, check=True)
    print(f"[{name}] converting {src} -> f16 gguf")
    conv_script = WORK / "Strata/build/_deps/strata_llamacpp-src/convert_hf_to_gguf.py"
    sh(["python3", str(conv_script), str(src), "--outfile", out, "--outtype", "f16"])
    q4 = GGUF_DIR / f"{name}.gguf"
    sh([str(LLAMA / "llama-quantize"), str(out), q4, "Q4_K_M"])
    if out.exists() and q4.exists():
        out.unlink()
    return q4 if q4.exists() else None


def bench(name):
    g = GGUF_DIR / f"{name}.gguf"
    assert g.exists(), f"fetch {name} first"
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%m%d-%H%M")
    out = RESULTS / f"bench_{name}_{tag}.json"
    r = sh([str(LLAMA / "llama-bench"), "-m", g, "-p", "128,2048,8192", "-n", "128",
            "-r", "3", "-t", "8", "-o", "json"], capture_output=True, text=True)
    out.write_text(r.stdout or r.stderr)
    print(f"[{name}] bench -> {out}")
    print(r.stdout[:2000])


def eval_model(name):
    g = GGUF_DIR / f"{name}.gguf"
    assert g.exists(), f"fetch {name} first"
    srv = subprocess.Popen([str(LLAMA / "llama-server"), "-m", g, "--port", "8091",
                            "-c", "8192", "-t", "8", "-ngl", "99", "--no-warmup"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    try:
        py = os.environ.get("PY310", str(Path.home() / "miniconda3/envs/py310/bin"))
        RESULTS.mkdir(parents=True, exist_ok=True)
        tag = time.strftime("%m%d-%H%M")
        out = RESULTS / f"quality_{name}_{tag}.json"
        sh([py + "/python", str(ROOT / "scripts/bench_quality.py"),
            "--base-url", "http://127.0.0.1:8091/v1", "--model", "zoo",
            "--data", str(WORK / "data"), "--n", "100", "--out", out])
    finally:
        srv.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch", "bench", "eval", "all"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in ZOO.items():
            print(f"{k:18s} {v['note']}")
        return
    name = a.name
    assert name in ZOO, f"unknown model {name}; see list"
    if a.cmd == "fetch":
        fetch(name)
    elif a.cmd == "bench":
        bench(name)
    elif a.cmd == "eval":
        eval_model(name)
    elif a.cmd == "all":
        g = fetch(name)
        if g:
            bench(name)
            eval_model(name)
            # delete weights ONLY if the eval wrote results, else keep for a re-run
            wrote = sorted(RAW.glob(f"quality_{name}_*.json"))
            if not a.keep and wrote:
                print(f"[{name}] deleting weights (results kept)")
                g.unlink(missing_ok=True)
            elif not wrote:
                print(f"[{name}] eval produced no results - keeping weights for re-run")


if __name__ == "__main__":
    main()
