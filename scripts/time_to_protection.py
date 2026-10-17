"""What the honeypot's intel adds on top of the model -> eval/results/protection.json.

python -m scripts.time_to_protection --part offline   # service stopped (memory)
python -m scripts.time_to_protection --part live      # against a fresh forest-api

Counterfactual replay of the test days (Sep 9-10). The scam handles in aliases.yaml are mapped
to the 20 accounts that receive the most laundering payments. We assume the honeypot reports
each account `delay` minutes after the first laundering payment reaches it, and replay the test
stream twice: model only, and model + intel from that moment on. A simulation with a stated
assumption, not an observed result.

--part live times the real loop on the running service: a scripted scammer session, the moment
the first id is stored, and the first victim payment that is held because of it.
"""

import argparse
import json
import time
from pathlib import Path

import httpx

from forest.config import DATA_DIR
from forest.guard.features import to_epoch
from forest.guard.score import Guard
from forest.honeypot.scenario import load_all
from forest.honeypot.simulator import Simulator
from scripts.make_aliases import mule_candidates
from scripts.replay import load_rows, seed_victim, victim_payment

OUT = Path(__file__).resolve().parent.parent / "eval/results/protection.json"
FIELDS = ("ts", "src", "dst", "from_bank", "to_bank", "amount", "currency", "fmt", "currency_recv")


class Activation:
    """Intel that switches on per account at a given dataset time."""

    def __init__(self, at: dict[str, int]) -> None:
        self.at = at
        self.now = 0

    def is_flagged(self, dst: str | None = None, handle: str | None = None) -> bool:
        t = self.at.get(dst) if dst else None
        return t is not None and self.now >= t


def replay(guard: Guard, rows: list[dict], intel: Activation | None) -> list[str]:
    out = []
    for r in rows:
        if intel is not None:
            intel.now = to_epoch(r["ts"])
        out.append(guard.score({k: r[k] for k in FIELDS}, rollback_stopped=True).decision)
    return out


def inr(guard: Guard, r: dict) -> float:
    return r["amount"] * guard.fx[r["currency"]] / guard.fx["Rupee"]


def simulate(rows: list[dict], targets: list[str], delay_min: int, base: list[str]) -> dict:
    first: dict[str, int] = {}
    for r in rows:
        if r["label"] == 1 and r["dst"] in targets and r["dst"] not in first:
            first[r["dst"]] = to_epoch(r["ts"])
    at = {a: t + delay_min * 60 for a, t in first.items()}
    guard = Guard(intel=Activation(at))
    with_intel = replay(guard, rows, guard.intel)

    stats = {k: 0 for k in ("laundering_after", "model_held", "model_paused", "extra_held")}
    rupees = {"laundering_after": 0.0, "model_held": 0.0, "extra_held": 0.0}
    benign = {"payments_held": 0, "rupees_held": 0.0}
    for r, a, b in zip(rows, base, with_intel):
        t = at.get(r["dst"])
        if t is None or to_epoch(r["ts"]) < t:
            continue
        v = inr(guard, r)
        if r["label"] == 1:
            stats["laundering_after"] += 1
            rupees["laundering_after"] += v
            if a == "hold":
                stats["model_held"] += 1
                rupees["model_held"] += v
            elif a == "pause":
                stats["model_paused"] += 1
            if b == "hold" and a != "hold":
                stats["extra_held"] += 1
                rupees["extra_held"] += v
        elif b == "hold" and a != "hold":
            benign["payments_held"] += 1
            benign["rupees_held"] += v
    return {
        "delay_minutes": delay_min,
        "accounts_with_laundering": len(first),
        **stats,
        **{f"{k}_inr": round(v) for k, v in rupees.items()},
        "benign_collateral": {**benign, "rupees_held": round(benign["rupees_held"])},
    }


def live_probe(url: str, scenario: str, rows: list[dict]) -> dict:
    """Scripted scammer chat -> id stored -> a victim payment to that account is held."""
    sc = load_all()[scenario]
    handle = sc["collect"][0]["value"]
    dst = json.loads((DATA_DIR / "aliases.json").read_text())[handle]
    ts = rows[0]["ts"]
    sim = Simulator(sc, 7)
    with httpx.Client(base_url=url, timeout=300) as c:
        seed_victim(c, url, ts)
        before = c.post("/score", json=victim_payment(ts, dst, handle, 85_000)).json()
        began = time.perf_counter()
        sid = c.post("/honeypot/sessions", json={"scenario": scenario}).raise_for_status()
        sid = sid.json()["id"]
        text, stored_at, turns = sim.opening(), None, 0
        while text is not None:
            out = c.post(f"/honeypot/sessions/{sid}/turn", json={"text": text}).json()
            turns += 1
            if stored_at is None and any(i["hits"] >= 0 for i in c.get("/intel").json()):
                stored_at = time.perf_counter() - began
                stored_turn = turns
                break
            if out["ended"]:
                break
            text = sim.respond(out["reply"])
        if stored_at is None:
            return {"error": "no intel stored", "before_decision": before["decision"]}
        t0 = time.perf_counter()
        after = c.post("/score", json=victim_payment(ts, dst, handle, 85_000)).json()
        probe_ms = (time.perf_counter() - t0) * 1000
    return {
        "scenario": scenario,
        "victim_payment_inr": 85_000,
        "before_intel": {
            "decision": before["decision"],
            "reasons": [r["code"] for r in before["reasons"]],
        },
        "after_intel": {
            "decision": after["decision"],
            "reasons": [r["code"] for r in after["reasons"]],
        },
        "seconds_from_first_scammer_message_to_id_stored": round(stored_at, 1),
        "turns_until_stored": stored_turn,
        "seconds_from_id_stored_to_hold_response": round(probe_ms / 1000, 4),
        "note": "chat time uses the real 350M model on this VM; a real scammer types slower",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", type=int, nargs="+", default=[30, 60, 240])
    ap.add_argument("--part", choices=["live", "offline"], required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--scenario", default="kyc_bank")
    args = ap.parse_args()

    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.part == "live":
        result["live"] = live_probe(args.url, args.scenario, load_rows(None, 10))
        print(json.dumps(result["live"], indent=2))
        OUT.write_text(json.dumps(result, indent=2))
        return

    rows = load_rows(None, None)
    targets = mule_candidates(20)
    print(f"{len(rows):,} test payments, {len(targets)} mapped accounts")
    result["assumption"] = (
        "intel arrives `delay_minutes` after the first laundering payment reaches the "
        "account; simulated, not observed"
    )
    base_guard = Guard()
    base = replay(base_guard, rows, None)
    laundering = [(r, d) for r, d in zip(rows, base) if r["label"] == 1]
    result["model_only"] = {
        "laundering_payments": len(laundering),
        "held": sum(d == "hold" for _, d in laundering),
        "paused": sum(d == "pause" for _, d in laundering),
        "laundering_inr": round(sum(inr(base_guard, r) for r, _ in laundering)),
        "held_inr": round(sum(inr(base_guard, r) for r, d in laundering if d == "hold")),
    }
    print(result["model_only"])
    result["with_intel"] = []
    for d in args.delays:
        s = simulate(rows, targets, d, base)
        result["with_intel"].append(s)
        print(s)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
