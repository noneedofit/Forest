"""Score one payment: features, Snap ML probability, then allow / pause / hold with reasons."""

import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from snapml import BoostingMachineClassifier

from forest.config import DATA_DIR
from forest.guard.features import FEATURES, LAST, to_epoch
from forest.guard.reasons import FALLBACK, PRIORITY

REQUIRED = ("ts", "src", "dst", "from_bank", "to_bank", "amount", "currency", "fmt")


class IntelLookup(Protocol):
    def is_flagged(self, dst: str | None = None, handle: str | None = None) -> bool: ...


@dataclass
class Decision:
    decision: str
    score: float
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def load_model(path: Path, tree_format: str = "auto") -> BoostingMachineClassifier:
    """Import the exported PMML. "auto" picks the zDNN tensor format when the AI accelerator
    is available, else compressed trees on the CPU."""
    model = BoostingMachineClassifier(n_jobs=1)
    model.import_model(str(path), "pmml", tree_format=tree_format)
    return model


def decide(p: float, t_pause: float, t_hold: float, intel_hit: bool) -> str:
    if intel_hit or p >= t_hold:
        return "hold"
    if p >= t_pause:
        return "pause"
    return "allow"


def reason_codes(f: dict[str, float], intel_hit: bool = False) -> list[str]:
    """Rule-based reasons from the features (not model explanations), top 3 by priority."""
    hits = {
        "INTEL_MATCH": intel_hit,
        "MULE_FANIN": f["r_n_new_senders_24h"] >= 5,
        "FAST_FORWARD": f["r_out_in_ratio_24h"] > 0.8,
        "AMOUNT_UNUSUAL": f["amt_z"] > 3,
        "NEW_RECEIVER": f["r_age_hours"] < 24,
        "NEW_PAYEE": f["s_new_payee"] == 1,
        "BURST": f["s_n_last_1h"] >= 3,
    }
    return [code for code in PRIORITY if hits[code]][:3]


class Guard:
    """Live scorer. Holds the feature state, so every call records the payment, whatever the
    decision. Payments must arrive in time order; to replay from the start, make a new Guard."""

    def __init__(
        self, artifacts_dir: Path = DATA_DIR / "artifacts", intel: IntelLookup | None = None
    ) -> None:
        if json.loads((artifacts_dir / "feature_list.json").read_text()) != FEATURES:
            raise ValueError("feature_list.json does not match the feature code; retrain")
        th = json.loads((artifacts_dir / "thresholds.json").read_text())
        self.t_pause = th["t_pause"]
        self.t_hold = th["t_hold"]
        self.fx = json.loads((artifacts_dir / "fx_rates.json").read_text())
        self.model = load_model(artifacts_dir / "guard_model.pmml")
        self.tree_format = self.model.booster_.optimized_tree_format_
        # Without the end-of-validation history every payment looks new and the thresholds
        # mean nothing, so a missing snapshot is an error, not a cold start.
        with open(artifacts_dir / "feature_state.pkl", "rb") as f:
            self.state = pickle.load(f)
        self.clock = max((rec[LAST] for rec in self.state.accts if rec), default=-1)
        self.intel = intel
        self._lock = threading.Lock()

    def score(self, txn: dict) -> Decision:
        """`txn` needs REQUIRED, with dataset names for currency ("Rupee") and fmt ("ACH").
        Optional: currency_recv (defaults to currency), dst_handle (UPI alias for intel)."""
        start = time.perf_counter()
        missing = [k for k in REQUIRED if txn.get(k) is None]
        if missing:
            raise ValueError(f"payment is missing {missing}")
        if txn["currency"] not in self.fx:
            raise ValueError(f"unknown currency {txn['currency']!r}")
        t = to_epoch(txn["ts"])
        row = {**txn, "ts": t, "amount_usd": float(txn["amount"]) * self.fx[txn["currency"]]}
        with self._lock:
            # The 24 h windows only move forward; an older payment would corrupt them.
            if t < self.clock:
                raise ValueError("payment is older than the last one scored; restart the Guard")
            self.clock = t
            feats = self.state.update_and_featurise(row)
            p = float(self.model.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
        intel_hit = self.intel is not None and self.intel.is_flagged(
            dst=txn["dst"], handle=txn.get("dst_handle")
        )
        decision = decide(p, self.t_pause, self.t_hold, intel_hit)
        reasons = []
        if decision != "allow":
            reasons = reason_codes(dict(zip(FEATURES, feats)), intel_hit) or [FALLBACK]
        return Decision(decision, p, reasons, (time.perf_counter() - start) * 1000)
