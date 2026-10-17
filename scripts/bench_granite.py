"""Speed, memory and sample replies for each Granite model on this machine.

python -m scripts.bench_granite [--models a.gguf b.gguf ...] [--out eval/results/granite_bench.json]

Each model runs in its own process (peak RSS is per process) through the same four scammer
turns with Kamla's real prompts, and no time budget.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from forest.honeypot import persona
from forest.honeypot.llm import N_THREADS, Granite

MODELS = [
    "/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf",
    "/models/granite-4.0-h-350m-Q8_0.gguf",
    "/models/granite-4.0-h-1b-bf16-be.gguf",
    "/models/granite-4.0-h-1b-Q8_0.gguf",
    "/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf",
]
TURNS = [
    ("Madam main Mumbai Cyber Cell ka officer hoon, ek zaroori case ke baare mein call kiya hai.",
     "confused_engage", None),
    (("Your Aadhaar is linked to a money laundering case. A warrant is ready and you will be "
      "arrested within two hours."), "scared_comply", None),
    ("Please read out the OTP you just received, madam. It is only for the verification.",
     "stall_credential", None),
    ("Transfer the verification amount of Rs 49,999 to inspector.verify22@okaxis immediately.",
     "confirm_and_ask_alt", "inspector.verify22@okaxis"),
]  # fmt: skip


def bench_one(path: str) -> dict:
    t = time.perf_counter()
    g = Granite(path, budget=float("inf"))
    load_s = time.perf_counter() - t
    history: list[dict] = []
    turns = []
    for text, move, echo in TURNS:
        history.append({"role": "scammer", "text": text})
        if g.chat:
            prompt = persona.chat_messages(history, move, echo)
            n_prompt = len(g.llm.tokenize(json.dumps(prompt).encode()))
            gen = g.generate(prompt)
            raw = gen.text
        else:
            prompt = persona.completion_prompt(history, move, echo)
            n_prompt = len(g.llm.tokenize(prompt.encode()))
            gen = g.generate(prompt)
            raw = (f" {echo}..." if echo else "") + gen.text
        history.append({"role": "agent", "text": raw.strip(), "raw": raw, "move": move})
        turns.append(
            {
                "move": move,
                "prompt_tokens": n_prompt,
                "seconds": round(gen.seconds, 2),
                "tokens": gen.tokens,
                "decode_tok_per_s": round(gen.tok_per_s, 2),
                "reply": raw.strip(),
            }
        )
    secs = [t["seconds"] for t in turns]
    return {
        "model": Path(path).name,
        "mode": "chat" if g.chat else "completion",
        "n_threads": N_THREADS,
        "load_s": round(load_s, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "first_reply_s": secs[0],
        "later_reply_s_mean": round(sum(secs[1:]) / len(secs[1:]), 2),
        "decode_tok_per_s_mean": round(sum(t["decode_tok_per_s"] for t in turns) / len(turns), 2),
        "turns": turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=Path("eval/results/granite_bench.json"))
    args = ap.parse_args()
    if args.one:
        print(json.dumps(bench_one(args.one)))
        return
    results = []
    for path in args.models:
        if not Path(path).exists():
            print(f"skip {path}: not found")
            continue
        print(f"== {path}", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.bench_granite", "--one", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            # -9 is the kernel's OOM killer on this 8 GB VM
            print(f"   failed with exit {proc.returncode}")
            results.append({"model": Path(path).name, "error": f"exit {proc.returncode}"})
            continue
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        print(
            f"   load {r['load_s']}s, peak {r['peak_rss_gb']} GB, first reply {r['first_reply_s']}s,"
            f" later {r['later_reply_s_mean']}s, decode {r['decode_tok_per_s_mean']} tok/s"
        )
        results.append(r)
    args.out.write_text(json.dumps({"machine": "IBM z17, 3 vCPU", "models": results}, indent=2))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
