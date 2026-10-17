import base64
import os

import pytest

from forest.honeypot.agent import Agent, cue_label
from forest.honeypot.scenario import load_all
from forest.honeypot.simulator import play
from forest.intel.store import Store

SCENARIOS = load_all()
REPORTED = {sid for sid, sc in SCENARIOS.items() if sc["reported"]}


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("forest.config.HMAC_KEY", "test-hmac-key")
    monkeypatch.setattr("forest.config.ENC_KEY", base64.b64encode(os.urandom(32)).decode())
    return Store(":memory:")


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_template_session_harvests_and_flags(store, sid):
    events = []
    agent = Agent(store, publish=lambda e, d: events.append(e), seed=3)
    s = play(agent, SCENARIOS[sid], REPORTED, seed=3)
    assert s.state.ids, "no collection id harvested"
    assert s.leaked == 0
    first = SCENARIOS[sid]["collect"][0]["value"]
    assert store.is_flagged(handle=first)
    assert "intel" in events and "chat" in events
    assert len(store.messages(s.id)) == 2 * s.state.turns


def test_unreported_sender_gets_no_session(store):
    with pytest.raises(PermissionError):
        Agent(store).start("someone", REPORTED)


def test_ended_session_takes_no_more_turns(store):
    agent = Agent(store, seed=1)
    s = play(agent, SCENARIOS["digital_arrest"], REPORTED, seed=1)
    with pytest.raises(ValueError):
        agent.turn(s, "hello?")


def test_cue_rules():
    assert cue_label(["CBI ka warrant hai aapke naam"]) == "digital_arrest"
    assert cue_label(["Registration fee for the internship"]) == "job_task"
    assert cue_label(["Nani, accident ho gaya, hospital mein hoon"]) == "family_emergency"
    assert cue_label(["Aapka KYC pending hai"]) == "kyc_bank"
    assert cue_label(["hello"]) is None
