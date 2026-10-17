import json
import pickle

import numpy as np
import pytest

from forest.guard.features import FEATURES, FeatureState
from forest.guard.load import FX_TO_USD
from forest.guard.reasons import FALLBACK, PRIORITY, TEXT, explain
from forest.guard.score import Guard, decide, reason_codes


def quiet() -> dict[str, float]:
    f = dict.fromkeys(FEATURES, 0.0)
    f["r_age_hours"] = 500.0
    return f


def test_decide_policy():
    assert decide(0.1, 0.5, 0.9, intel_hit=False) == "allow"
    assert decide(0.5, 0.5, 0.9, intel_hit=False) == "pause"
    assert decide(0.95, 0.5, 0.9, intel_hit=False) == "hold"
    assert decide(0.0, 0.5, 0.9, intel_hit=True) == "hold"


def test_reasons_none_when_quiet():
    assert reason_codes(quiet()) == []


def test_reasons_capped_at_three_in_priority_order():
    f = quiet()
    f.update(s_new_payee=1, amt_z=4.0, s_n_last_1h=5, r_age_hours=2.0, r_n_new_senders_24h=6)
    assert reason_codes(f) == ["MULE_FANIN", "AMOUNT_UNUSUAL", "NEW_RECEIVER"]
    assert reason_codes(f, intel_hit=True)[0] == "INTEL_MATCH"


def test_every_code_has_both_languages():
    assert set(TEXT) == {*PRIORITY, FALLBACK}
    assert explain(["NEW_PAYEE"], "hi") == [
        "Aapne is khaate mein pehle kabhi paise nahi bheje hain."
    ]


class FlagOne:
    def __init__(self, bad: str):
        self.bad = bad

    def is_flagged(self, dst: str | None = None, handle: str | None = None) -> bool:
        return self.bad in (dst, handle)


@pytest.fixture
def artifacts(tmp_path):
    from snapml import BoostingMachineClassifier

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(FEATURES))).astype(np.float32)
    y = (x[:, FEATURES.index("log_amt")] > 1.0).astype(np.int8)
    model = BoostingMachineClassifier(num_round=20, max_depth=3, random_state=0).fit(x, y)
    model.export_model(str(tmp_path / "guard_model.pmml"), "pmml")
    (tmp_path / "feature_list.json").write_text(json.dumps(FEATURES))
    (tmp_path / "fx_rates.json").write_text(json.dumps(FX_TO_USD))
    (tmp_path / "thresholds.json").write_text(json.dumps({"t_pause": 0.5, "t_hold": 0.99}))
    (tmp_path / "feature_state.pkl").write_bytes(pickle.dumps(FeatureState()))
    return tmp_path


TXN = {
    "ts": "2022/09/12 10:00",
    "src": "A",
    "dst": "B",
    "from_bank": "001",
    "to_bank": "002",
    "amount": 20,
    "currency": "US Dollar",
    "fmt": "ACH",
}


def test_guard_scores_and_holds_intel_hits(artifacts):
    guard = Guard(artifacts, intel=FlagOne("MULE1"))
    txn = TXN
    d = guard.score(txn)
    assert d.decision in {"allow", "pause", "hold"}
    assert 0.0 <= d.score <= 1.0
    assert d.latency_ms > 0

    d = guard.score({**txn, "dst": "MULE1"})
    assert d.decision == "hold"
    assert d.reasons[0] == "INTEL_MATCH"

    d = guard.score({**txn, "dst": "C", "dst_handle": "MULE1"})
    assert d.decision == "hold"


def test_guard_rejects_bad_payments(artifacts):
    guard = Guard(artifacts)
    with pytest.raises(ValueError, match="currency"):
        guard.score({**TXN, "currency": "INR"})
    with pytest.raises(ValueError, match="missing"):
        guard.score({k: v for k, v in TXN.items() if k != "fmt"})
    guard.score(TXN)
    with pytest.raises(ValueError, match="older"):
        guard.score({**TXN, "ts": "2022/09/12 09:59"})
    assert guard.score(TXN).decision in {"allow", "pause", "hold"}  # same minute is fine


def test_guard_needs_warm_state(artifacts):
    (artifacts / "feature_state.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        Guard(artifacts)


def test_model_only_stop_gets_fallback_reason(artifacts, monkeypatch):
    monkeypatch.setattr("forest.guard.score.reason_codes", lambda f, intel_hit=False: [])
    guard = Guard(artifacts)
    guard.t_pause = guard.t_hold = 0.0
    d = guard.score(TXN)
    assert d.decision == "hold"
    assert d.reasons == [FALLBACK]
