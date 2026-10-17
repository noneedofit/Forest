import base64
import os

import pytest

from forest.intel.store import Store


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("forest.config.HMAC_KEY", "test-hmac-key")
    monkeypatch.setattr("forest.config.ENC_KEY", base64.b64encode(os.urandom(32)).decode())
    return Store(":memory:")


def test_intel_stores_hash_and_mask_only(store):
    row = store.add_intel("ramesh.k22@okaxis", "upi", "digital_arrest", "s1", 0.9)
    assert row["display"] == "ra****@okaxis"
    assert "ramesh" not in str(store.list_intel())
    assert store.is_flagged(handle="ramesh.k22@okaxis")
    assert not store.is_flagged(handle="someone.else@okaxis")


def test_adding_twice_keeps_one_row(store):
    store.add_intel("a.b@okaxis", "upi")
    store.add_intel("a.b@okaxis", "upi")
    assert len(store.list_intel()) == 1


def test_handle_flags_its_alias_account(store):
    store.add_alias("inspector.verify22@okaxis", "8000EBD30")
    assert not store.is_flagged(dst="8000EBD30")
    store.add_intel("inspector.verify22@okaxis", "upi", "digital_arrest")
    assert store.is_flagged(dst="8000EBD30")
    hit = store.match(dst="8000EBD30")
    store.add_hit(hit)
    assert store.list_intel()[0]["hits"] == 1


def test_flagged_set_survives_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr("forest.config.HMAC_KEY", "k")
    path = tmp_path / "f.db"
    s = Store(path)
    s.add_alias("x.y@okaxis", "ACC1")
    s.add_intel("x.y@okaxis", "upi")
    assert Store(path).is_flagged(dst="ACC1")


def test_messages_are_encrypted_at_rest(store):
    store.add_message("s1", "scammer", "send money to ramesh.k22@okaxis")
    raw = store.db.execute("SELECT body_enc FROM messages").fetchone()[0]
    assert b"ramesh" not in raw
    assert store.messages("s1")[0]["body"] == "send money to ramesh.k22@okaxis"


def decision(amount: float, kind: str, reasons: str, label: int, victim: int = 0) -> dict:
    return {
        "ts": "t",
        "src_hash": "a",
        "dst_hash": "b",
        "dst_display": "x",
        "score": 0.9,
        "latency_ms": 0.1,
        "amount_inr": amount,
        "decision": kind,
        "reasons": reasons,
        "label": label,
        "victim": victim,
    }


def test_totals(store):
    store.log_decision(decision(100.0, "hold", "MULE_FANIN", 1))
    store.log_decision(decision(50.0, "hold", "INTEL_MATCH", 0, victim=1))
    store.log_decision(decision(70.0, "hold", "INTEL_MATCH", 0))
    store.log_decision(decision(5.0, "allow", "", 0))
    t = store.decision_totals()
    assert (t["scored"], t["held"], t["held_by_intel"]) == (4, 3, 2)
    assert t["inr_protected"] == 150.0
