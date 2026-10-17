"""Run a simulated scammer conversation against Kamla and print the transcript with timings.

python -m scripts.run_session --scenario digital_arrest            # in-process, scratch DB
python -m scripts.run_session --scenario digital_arrest --demo     # through the API, so the dashboard animates
python -m scripts.run_session --scenario all --templates           # no Granite, hand-written lines only
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import httpx

from forest.honeypot import llm
from forest.honeypot.agent import Agent
from forest.honeypot.scenario import load_all
from forest.honeypot.simulator import Simulator, play
from forest.intel.store import Store


def show(text: str, out: dict) -> None:
    ents = ", ".join(
        f"{e['kind']}:{e['normalized']}" for e in out["extracted"] if e["kind"] != "amount"
    )
    print(f"\nSCAMMER [{out['stage']}] {text}")
    if ents:
        print(f"   extracted: {ents}")
    print(f"KAMLA   [{out['move']}, {out['source']}, {out['seconds']:.1f}s] {out['reply']}")


def record(turns: list[dict], text: str, out: dict) -> None:
    turns.append({"scammer": text, **out})
    show(text, out)


def run_local(scenarios: list[dict], args) -> list[dict]:
    granite = None if args.templates else llm.load(args.model)
    print(f"model: {granite.name if granite else 'templates only'}")
    agent = Agent(Store(":memory:"), granite, seed=args.seed)
    reported = {sc["id"] for sc in load_all().values() if sc["reported"]}
    results = []
    for sc in scenarios:
        print(f"\n=== {sc['id']} ({sc['caller']})")
        turns: list[dict] = []
        began = time.perf_counter()
        s = play(agent, sc, reported, seed=args.seed, on_turn=partial(record, turns))
        report = agent.report(s)
        print(f"\nreport ({report['summary_by']}): {report['summary']}")
        print(
            f"turns {s.state.turns}, ids {len(s.state.ids)}, scammer minutes "
            f"{s.state.scammer_seconds / 60:.1f}, wall {time.perf_counter() - began:.0f}s"
        )
        results.append(
            {
                "scenario": sc["id"],
                "model": granite.name if granite else None,
                "turns": turns,
                "replies": s.replies,
                "ids": s.state.ids,
                "scammer_seconds": s.state.scammer_seconds,
                "leaked": s.leaked,
                "report": report,
            }
        )
    return results


def run_api(scenarios: list[dict], args) -> list[dict]:
    results = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        for sc in scenarios:
            print(f"\n=== {sc['id']} ({sc['caller']})")
            sid = client.post("/honeypot/sessions", json={"scenario": sc["id"]}).raise_for_status()
            sid = sid.json()["id"]
            sim = Simulator(sc, args.seed)
            text = sim.opening()
            turns = []
            while text is not None:
                out = client.post(f"/honeypot/sessions/{sid}/turn", json={"text": text})
                out = out.raise_for_status().json()
                turns.append({"scammer": text, **out})
                show(text, out)
                if out["ended"]:
                    break
                time.sleep(args.pace)
                text = sim.respond(out["reply"])
            report = client.get(f"/honeypot/sessions/{sid}/report").raise_for_status().json()
            print(f"\nreport ({report['summary_by']}): {report['summary']}")
            results.append({"scenario": sc["id"], "session": sid, "turns": turns, "report": report})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="digital_arrest", help="scenario id, or 'all'")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--demo", action="store_true", help="go through the running API")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between turns in --demo")
    ap.add_argument("--model", help="GGUF path (default FOREST_MODEL_PATH)")
    ap.add_argument("--templates", action="store_true", help="hand-written lines only")
    ap.add_argument("--out", type=Path, help="write the transcripts here as JSON")
    args = ap.parse_args()

    all_sc = load_all()
    scenarios = list(all_sc.values()) if args.scenario == "all" else [all_sc[args.scenario]]
    results = run_api(scenarios, args) if args.demo else run_local(scenarios, args)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
