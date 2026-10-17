"""Scoring latency and throughput on the LinuxONE VM -> eval/results/latency.json.

python -m scripts.bench_latency            # needs a freshly started forest-api (start_demo.sh --fresh)

Three parts:
  inproc   Guard.score() called directly, 1 thread, then 1 vs 3 independent replicas
  http     /score on the running service with the honeypot idle, then while a 350M session is
           generating replies, then while the 2B model writes the incident summary
  platform machine type, CPUs, library versions

Each HTTP phase runs 1 client and 3 clients. Clients send in lock-step groups that share a
timestamp, because the service refuses a payment older than the last one it scored.
"""

import argparse
import json
import os
import platform
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import httpx
import numpy as np

from forest.guard.score import Guard
from forest.honeypot.scenario import load_all
from forest.honeypot.simulator import Simulator
from scripts.replay import load_rows

OUT = Path(__file__).resolve().parent.parent / "eval/results/latency.json"
SCORE_FIELDS = ("ts", "src", "dst", "from_bank", "to_bank", "amount", "currency", "fmt")


def pct(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "n": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "max_ms": round(float(a.max()), 3),
    }


def body(row: dict) -> dict:
    return {k: row[k] for k in (*SCORE_FIELDS, "currency_recv")}


def platform_info() -> dict:
    cpu = Path("/proc/cpuinfo").read_text()
    sysinfo = Path("/proc/sysinfo").read_text() if Path("/proc/sysinfo").exists() else ""
    machine = re.search(r"machine\s*=\s*(\d+)", cpu)
    hyper = re.search(r"VM00 Control Program:\s*(.+)", sysinfo)
    import snapml

    return {
        "machine_type": machine.group(1) if machine else None,
        "nnpa": re.search(r"^features\s*:.*\bnnpa\b", cpu, re.MULTILINE) is not None,
        "vcpus": os.cpu_count(),
        "control_program": hyper.group(1).strip() if hyper else None,
        "python": platform.python_version(),
        "snapml": getattr(snapml, "__version__", None) or "1.17.2",
        "arch": platform.machine(),
    }


def inproc_latency(rows: list[dict], n: int, warm: int) -> dict:
    guard = Guard()
    for r in rows[:warm]:
        guard.score(body(r), rollback_stopped=True)
    lat = [guard.score(body(r), rollback_stopped=True).latency_ms for r in rows[warm : warm + n]]
    return {"tree_format": guard.tree_format, **pct(lat)}


def replica(args: tuple[list[dict], int]) -> tuple[int, float]:
    """One independent Guard on its own slice: how fast does a copy run when others share the CPUs."""
    rows, warm = args
    guard = Guard()
    for r in rows[:warm]:
        guard.score(body(r), rollback_stopped=True)
    start = time.perf_counter()
    for r in rows[warm:]:
        guard.score(body(r), rollback_stopped=True)
    return len(rows) - warm, time.perf_counter() - start


def inproc_scaling(rows: list[dict], per_worker: int, warm: int) -> list[dict]:
    out = []
    for workers in (1, 3):
        # separate slices of the test stream; each replica starts from the Sep 8 state
        slices = [
            (rows[i * (per_worker + warm) : (i + 1) * (per_worker + warm)], warm)
            for i in range(workers)
        ]
        began = time.perf_counter()
        with Pool(workers) as pool:
            done = pool.map(replica, slices)
        wall = time.perf_counter() - began
        rate = sum(n / s for n, s in done)
        out.append(
            {
                "replicas": workers,
                "scored_per_replica": per_worker,
                "payments_per_s_total": round(rate),
                "payments_per_s_each": round(rate / workers),
                "wall_s": round(wall, 1),
            }
        )
        print(f"  inproc {workers} replica(s): {rate:,.0f}/s total")
    return out


class Feed:
    """Hands out payments in time order, `clients` at a time, all with the same timestamp."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.pos = 0

    def take(self, k: int) -> list[dict]:
        grp = self.rows[self.pos : self.pos + k]
        self.pos += k
        ts = grp[-1]["ts"]
        return [{**body(r), "ts": ts} for r in grp]


def http_phase(url: str, feed: Feed, clients: int, tps: float | None, stop, min_n: int) -> dict:
    """Send until `stop()` is true and at least `min_n` payments went out. tps=None: flat out."""
    rtt: list[float] = []
    srv: list[float] = []
    errors = 0
    pool = ThreadPoolExecutor(clients)
    http = [httpx.Client(base_url=url, timeout=60) for _ in range(clients)]

    def one(i: int, b: dict) -> tuple[float, float] | None:
        t = time.perf_counter()
        r = http[i].post("/score", json=b)
        if r.status_code != 200:
            return None
        return (time.perf_counter() - t) * 1000, r.json()["latency_ms"]

    began = time.perf_counter()
    groups = 0
    while not (stop() and len(rtt) >= min_n):
        if tps:
            wait = began + groups * clients / tps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        groups += 1
        res = list(pool.map(lambda a: one(*a), enumerate(feed.take(clients))))
        for x in res:
            if x is None:
                errors += 1
            else:
                rtt.append(x[0])
                srv.append(x[1])
    wall = time.perf_counter() - began
    pool.shutdown()
    for c in http:
        c.close()
    return {
        "clients": clients,
        "target_tps": tps,
        "achieved_tps": round(len(rtt) / wall, 1),
        "errors": errors,
        "round_trip": pct(rtt),
        "guard_only": pct(srv),
    }


def run_session(url: str, scenario: str, seed: int, events: dict) -> None:
    """Drive one scammer script through the API; flags `events` as the phases begin and end."""
    sc = load_all()[scenario]
    sim = Simulator(sc, seed)
    with httpx.Client(base_url=url, timeout=300) as c:
        sid = c.post("/honeypot/sessions", json={"scenario": scenario}).raise_for_status()
        sid = sid.json()["id"]
        text = sim.opening()
        while text is not None:
            out = c.post(f"/honeypot/sessions/{sid}/turn", json={"text": text}).json()
            if out["ended"]:
                break
            text = sim.respond(out["reply"])
        events["chat_done"].set()
        # the 2B summary arrives later in the background
        while c.get(f"/honeypot/sessions/{sid}/report").json()["summary_by"] == "pending":
            time.sleep(2)
    events["summary_done"].set()


def loaded_phases(url: str, feed: Feed, clients: int, tps: float, scenario: str) -> dict:
    ev = {"chat_done": threading.Event(), "summary_done": threading.Event()}
    t = threading.Thread(target=run_session, args=(url, scenario, 7, ev))
    t.start()
    chat = http_phase(url, feed, clients, tps, ev["chat_done"].is_set, 200)
    # the chat's final turn ends the session; the summary task starts at once
    summary = http_phase(url, feed, clients, tps, ev["summary_done"].is_set, 200)
    t.join()
    return {"during_350m_session": chat, "during_2b_summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["inproc", "http"], required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tps", type=float, default=50, help="total replay rate for the paced phases")
    ap.add_argument("--inproc", type=int, default=20_000)
    ap.add_argument("--scale-n", type=int, default=20_000)
    ap.add_argument("--idle-n", type=int, default=5_000)
    ap.add_argument("--scenario", default="kyc_bank")
    args = ap.parse_args()

    # the replicas each load their own copy of the feature state, so run `inproc` with the
    # service stopped (an OOM kill took forest-api down when both ran at once)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["platform"] = platform_info()
    result["note"] = f"paced phases send {args.tps:g} payments/s in total"

    if args.part == "inproc":
        rows = load_rows(None, args.inproc + 1000 + 3 * (args.scale_n + 1000))
        print("in-process, 1 thread")
        result["inproc"] = inproc_latency(rows, args.inproc, 1000)
        print(f"  {result['inproc']}")
        result["inproc_replicas"] = inproc_scaling(rows, args.scale_n, 1000)
    else:
        feed = Feed(load_rows(None, 120_000))
        http: dict = {}
        for clients in (1, 3):
            print(f"http idle, {clients} client(s), flat out")
            http[f"idle_flat_out_{clients}"] = http_phase(
                args.url, feed, clients, None, lambda: True, args.idle_n
            )
            print(f"http idle, {clients} client(s), {args.tps:g} tps")
            http[f"idle_paced_{clients}"] = http_phase(
                args.url, feed, clients, args.tps, lambda: True, 1000
            )
        for clients in (1, 3):
            print(f"http under Granite load, {clients} client(s)")
            http[f"loaded_{clients}"] = loaded_phases(
                args.url, feed, clients, args.tps, args.scenario
            )
            print(f"  {json.dumps(http[f'loaded_{clients}'])[:400]}")
        result["http"] = http
        result["service_stats"] = httpx.get(f"{args.url}/stats").json()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
