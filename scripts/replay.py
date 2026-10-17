"""Stream the test-period payments into a running /score endpoint.

python -m scripts.replay --tps 50 --inject-victim 30

The guard state is the end of Sep 8, so the replay starts at the first Sep 9 row and the
service must be freshly started (it refuses payments older than the last one it scored).
"""

import argparse
import csv
import json
import time
from collections import Counter

import httpx

from forest.config import DATA_DIR
from forest.guard.features import to_epoch
from forest.guard.train import RAW

TEST_DAYS = ("2022/09/09", "2022/09/10")
VICTIM = "FRST-VICTIM-KAMLA"
SHOPS = ["FRST-SHOP-GROCER", "FRST-SHOP-PHARMACY", "FRST-SHOP-MILK", "FRST-SHOP-ELECTRIC"]
VICTIM_BANK = "0100"


def load_rows(start: str | None, limit: int | None) -> list[dict]:
    """Test-period payments as /score bodies, in time order."""
    floor = to_epoch(start) if start else 0
    rows = []
    with open(RAW, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if r[0].startswith(TEST_DAYS) and to_epoch(r[0]) >= floor:
                rows.append(
                    {
                        "ts": r[0],
                        "from_bank": r[1],
                        "src": r[2],
                        "to_bank": r[3],
                        "dst": r[4],
                        "currency_recv": r[6],
                        "amount": float(r[7]),
                        "currency": r[8],
                        "fmt": r[9],
                        "label": int(r[10]),
                    }
                )
    rows.sort(key=lambda x: x["ts"])
    return rows[:limit] if limit else rows


def victim_payment(ts: str, dst: str, handle: str, amount: float) -> dict:
    return {
        "ts": ts,
        "src": VICTIM,
        "dst": dst,
        "from_bank": VICTIM_BANK,
        "to_bank": "0200",
        "amount": amount,
        "currency": "Rupee",
        "fmt": "ACH",
        "dst_handle": handle,
        "victim": True,
    }


def seed_victim(client: httpx.Client, url: str, ts: str) -> None:
    """A few ordinary small payments so the victim looks like someone with a normal history."""
    for i in range(8):
        body = {
            "ts": ts,
            "src": VICTIM,
            "dst": SHOPS[i % len(SHOPS)],
            "from_bank": VICTIM_BANK,
            "to_bank": "0300",
            "amount": 400.0 + 150 * (i % 3),
            "currency": "Rupee",
            "fmt": "ACH",
        }
        client.post(f"{url}/score", json=body).raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tps", type=float, default=50)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--from", dest="start", help='first payment time, e.g. "2022/09/09 12:00"')
    ap.add_argument("--inject-victim", type=float, default=0, metavar="SECONDS")
    ap.add_argument("--victim-handle", help="always pay this handle (default: cycle through all)")
    ap.add_argument("--victim-amount", type=float, default=85_000, help="in rupees")
    args = ap.parse_args()

    aliases = {}
    if args.inject_victim:
        aliases = json.loads((DATA_DIR / "aliases.json").read_text())
    pairs = list(aliases.items())
    if args.victim_handle:
        pairs = [(args.victim_handle, aliases[args.victim_handle])]

    rows = load_rows(args.start, args.limit)
    print(f"{len(rows):,} payments, first {rows[0]['ts']}, last {rows[-1]['ts']}")
    counts: Counter[str] = Counter()
    errors = 0
    began = last_print = last_victim = time.time()
    victims = 0
    with httpx.Client(timeout=30) as client:
        if pairs:
            seed_victim(client, args.url, rows[0]["ts"])
        for i, row in enumerate(rows):
            wait = began + i / args.tps - time.time()
            if wait > 0:
                time.sleep(wait)
            r = client.post(f"{args.url}/score", json=row)
            if r.status_code == 200:
                counts[r.json()["decision"]] += 1
            elif r.status_code == 422:
                raise SystemExit(f"{r.json()['detail']} (restart the service for a fresh replay)")
            else:
                errors += 1
            now = time.time()
            if pairs and now - last_victim >= args.inject_victim:
                last_victim = now
                handle, dst = pairs[victims % len(pairs)]
                victims += 1
                body = victim_payment(row["ts"], dst, handle, args.victim_amount)
                v = client.post(f"{args.url}/score", json=body).json()
                print(f"  victim -> {handle}: {v['decision']} {[x['code'] for x in v['reasons']]}")
            if now - last_print >= 5:
                last_print = now
                done = i + 1
                print(
                    f"{done:,} sent, {done / (now - began):.0f}/s, {dict(counts)}, errors {errors}"
                )
    print(f"done: {dict(counts)}, errors {errors}")


if __name__ == "__main__":
    main()
