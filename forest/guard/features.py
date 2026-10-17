"""Streaming payment features.

Every feature is computed from transactions strictly before the current one, then the current
transaction is folded into the state. Training loops `FeatureState.step` over the whole file in
time order; live scoring calls `update_and_featurise` once per payment, which ends in the same
`step`, so the two can't drift apart.
"""

import calendar
import gc
import math
import sys
import time
from array import array
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY

FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
FMT_INDEX = {f: i for i, f in enumerate(FORMATS)}

FEATURES = [
    "s_n_prior",
    "s_mean_log_amt",
    "s_std_log_amt",
    "amt_z",
    "s_new_payee",
    "s_n_payees",
    "s_secs_since_last",
    "s_n_last_1h",
    "s_n_last_24h",
    "s_sum_last_24h_log",
    "r_n_senders_24h",
    "r_n_new_senders_24h",
    "r_in_log_24h",
    "r_age_hours",
    "r_out_in_ratio_24h",
    "log_amt",
    "hour",
    "dow",
    "is_self",
    "cross_bank",
    "cur_mismatch",
] + ["fmt_" + f.lower().replace(" ", "_") for f in FORMATS]

# Slots of the per-account record (a plain list keeps 500k accounts small). OUT_HEAD counts
# entries popped from OUT and H1 is the absolute index of the oldest one inside the last hour,
# so the 1 h count stays O(1) for accounts sending thousands of payments an hour.
N, MEAN, M2, LAST, PAYEES, FIRST, OUT, OUT_SUM, OUT_HEAD, H1, IN, IN_SUM, IN_SENDERS, IN_NEW = (
    range(14)
)


def to_epoch(ts: float | str | datetime) -> int:
    """Seconds since epoch, treating naive times as UTC (dataset time, not wall clock)."""
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if isinstance(ts, float):
        return int(ts)
    if isinstance(ts, str):
        try:
            ts = datetime.strptime(ts, "%Y/%m/%d %H:%M")
        except ValueError:
            ts = datetime.fromisoformat(ts)
    if ts.tzinfo is not None:
        return int(ts.timestamp())
    return calendar.timegm(ts.timetuple())


def _prune_out(rec: list, t: int) -> None:
    win = rec[OUT]
    if win is None:
        return
    while win and win[0][0] <= t - DAY:
        rec[OUT_SUM] -= win.popleft()[1]
        rec[OUT_HEAD] += 1
    if not win:
        rec[OUT_SUM] = 0.0


def _prune_in(rec: list, t: int) -> None:
    win = rec[IN]
    if win is None:
        return
    senders = rec[IN_SENDERS]
    while win and win[0][0] <= t - DAY:
        _, src, amt, new = win.popleft()
        rec[IN_SUM] -= amt
        left = senders[src] - 1
        if left:
            senders[src] = left
        else:
            del senders[src]
        rec[IN_NEW] -= new
    if not win:
        rec[IN_SUM] = 0.0


class FeatureState:
    """Per-account history needed to featurise the next payment."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.accts: list[list | None] = []
        self.pairs: set[int] = set()

    def account_id(self, account: str) -> int:
        i = self.ids.get(account)
        if i is None:
            i = len(self.accts)
            self.ids[account] = i
            self.accts.append(None)
        return i

    def _rec(self, i: int, t: int) -> list:
        rec = self.accts[i]
        if rec is None:
            rec = [0, 0.0, 0.0, -1, 0, t, None, 0.0, 0, 0, None, 0.0, None, 0]
            self.accts[i] = rec
        return rec

    def step(
        self,
        t: int,
        s: int,
        d: int,
        amt: float,
        fmt: int,
        cross_bank: bool,
        cur_mismatch: bool,
    ) -> list[float]:
        """Featurise one payment from account id `s` to `d`, then add it to the state."""
        srec = self._rec(s, t)
        drec = self._rec(d, t)
        log_amt = math.log1p(amt)

        n = srec[N]
        mean = srec[MEAN]
        std = math.sqrt(srec[M2] / n) if n > 1 else 0.0
        amt_z = (log_amt - mean) / max(std, 0.25) if n >= 3 else 0.0
        pair = (s << 32) | d
        new_payee = pair not in self.pairs
        secs = min(t - srec[LAST], WEEK) if srec[LAST] >= 0 else WEEK

        _prune_out(srec, t)
        out = srec[OUT]
        n_24h = len(out) if out else 0
        head = srec[OUT_HEAD]
        h1 = max(srec[H1], head)
        end = head + n_24h
        while h1 < end and out[h1 - head][0] <= t - HOUR:
            h1 += 1
        srec[H1] = h1
        n_1h = end - h1
        sum_24h_log = math.log1p(srec[OUT_SUM])

        _prune_in(drec, t)
        _prune_out(drec, t)
        n_senders = len(drec[IN_SENDERS]) if drec[IN_SENDERS] else 0
        in_sum = drec[IN_SUM]
        ratio = drec[OUT_SUM] / in_sum if in_sum > 0 else 0.0

        feats = [
            n,
            mean,
            std,
            amt_z,
            new_payee,
            srec[PAYEES],
            secs,
            n_1h,
            n_24h,
            sum_24h_log,
            n_senders,
            drec[IN_NEW],
            math.log1p(in_sum),
            (t - drec[FIRST]) / HOUR,
            ratio,
            log_amt,
            (t // HOUR) % 24,
            (t // DAY + 3) % 7,  # 1970-01-01 was a Thursday; Monday = 0
            s == d,
            cross_bank,
            cur_mismatch,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        if fmt >= 0:
            feats[21 + fmt] = 1

        n += 1
        delta = log_amt - mean
        mean += delta / n
        srec[N] = n
        srec[MEAN] = mean
        srec[M2] += delta * (log_amt - mean)
        srec[LAST] = t
        if new_payee:
            self.pairs.add(pair)
            srec[PAYEES] += 1
        if out is None:
            out = srec[OUT] = deque()
        out.append((t, amt))
        srec[OUT_SUM] += amt

        if drec[IN] is None:
            drec[IN] = deque()
            drec[IN_SENDERS] = {}
        drec[IN].append((t, s, amt, new_payee))
        drec[IN_SUM] += amt
        senders = drec[IN_SENDERS]
        senders[s] = senders.get(s, 0) + 1
        drec[IN_NEW] += new_payee
        return feats

    def update_and_featurise(self, txn: dict) -> list[float]:
        """Live path. `txn` needs ts, src, dst, amount_usd, and optionally from_bank, to_bank,
        currency, currency_recv, fmt."""
        cur = txn.get("currency")
        return self.step(
            to_epoch(txn["ts"]),
            self.account_id(txn["src"]),
            self.account_id(txn["dst"]),
            float(txn["amount_usd"]),
            FMT_INDEX.get(txn.get("fmt"), -1),
            txn.get("from_bank") != txn.get("to_bank"),
            cur != txn.get("currency_recv", cur),
        )


def prepare_aml(df: pd.DataFrame, state: FeatureState) -> dict[str, list]:
    """Map account strings to ids inside `state` and turn the columns into plain lists."""
    codes, uniques = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    for account in uniques:
        state.account_id(account)
    remap = np.array([state.ids[a] for a in uniques], dtype=np.int64)
    n = len(df)
    ids = remap[codes]
    fmt_codes = df["fmt"].map(FMT_INDEX).fillna(-1).astype(np.int64)
    return {
        "ts": df["ts"].to_numpy().tolist(),
        "src": ids[:n].tolist(),
        "dst": ids[n:].tolist(),
        "amt": df["amount_usd"].to_numpy(np.float64).tolist(),
        "fmt": fmt_codes.to_numpy().tolist(),
        "cross_bank": (
            df["from_bank"].astype(str).to_numpy() != df["to_bank"].astype(str).to_numpy()
        ).tolist(),
        "cur_mismatch": (
            df["cur_paid"].astype(str).to_numpy() != df["cur_recv"].astype(str).to_numpy()
        ).tolist(),
    }


def featurise_rows(
    state: FeatureState, cols: dict[str, list], start: int, end: int, every: int = 500_000
) -> np.ndarray:
    """Run `state.step` over rows [start, end) of `prepare_aml` output; float32 matrix."""
    out = array("f")
    step = state.step
    t0 = time.time()
    # Millions of small long-lived lists/deques make the cyclic GC rescan the heap over and
    # over; nothing here forms cycles, so switch it off for the pass.
    gc_was_on = gc.isenabled()
    gc.disable()
    rows = zip(
        cols["ts"][start:end],
        cols["src"][start:end],
        cols["dst"][start:end],
        cols["amt"][start:end],
        cols["fmt"][start:end],
        cols["cross_bank"][start:end],
        cols["cur_mismatch"][start:end],
    )
    for k, row in enumerate(rows, start=1):
        out.extend(step(*row))
        if k % every == 0:
            rate = k / (time.time() - t0)
            print(f"  {start + k:,} rows, {rate:,.0f} rows/s", file=sys.stderr, flush=True)
    if gc_was_on:
        gc.enable()
    return np.frombuffer(out, dtype=np.float32).reshape(-1, len(FEATURES))
