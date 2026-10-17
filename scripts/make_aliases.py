"""Map the fictitious scam handles in aliases.yaml onto mule-like receiver accounts.

python -m scripts.make_aliases [--count 20]

Picks the accounts that receive the most laundering payments in the test period, stores the
handle -> account link (hashed) in the database, and writes data/aliases.json for replay.py.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from forest.config import DATA_DIR, DB_PATH
from forest.guard.train import RAW
from forest.intel.store import Store

YAML = Path(__file__).resolve().parent.parent / "forest/honeypot/scenarios/aliases.yaml"
OUT = DATA_DIR / "aliases.json"
TEST_DAYS = ("2022/09/09", "2022/09/10")


def mule_candidates(n: int) -> list[str]:
    """Receivers of the most laundering payments on the test days."""
    counts: Counter[str] = Counter()
    with open(RAW, newline="") as f:
        rows = csv.reader(f)
        next(rows)
        for r in rows:
            if r[0].startswith(TEST_DAYS) and r[-1] == "1":
                counts[r[4]] += 1
    return [acct for acct, _ in counts.most_common(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()
    handles = yaml.safe_load(YAML.read_text())["handles"][: args.count]
    accounts = mule_candidates(len(handles))
    if len(accounts) < len(handles):
        raise SystemExit(f"only {len(accounts)} mule candidates for {len(handles)} handles")
    store = Store(DB_PATH)
    pairs = dict(zip(handles, accounts))
    for handle, account in pairs.items():
        store.add_alias(handle, account)
    OUT.write_text(json.dumps(pairs, indent=2))
    print(f"mapped {len(pairs)} handles, written to {OUT}")


if __name__ == "__main__":
    main()
