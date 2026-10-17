import pytest

from forest.honeypot.simulator import Simulator, asks_something
from forest.honeypot.strategy import MAX_TURNS, State, next_move, wasted_seconds


@pytest.mark.parametrize(
    "stage,move",
    [
        ("hook", "confused_engage"),
        ("authority_claim", "confused_engage"),
        ("smalltalk", "confused_engage"),
        ("threat", "scared_comply"),
        ("credential_ask", "stall_credential"),
        ("payment_ask", "ask_payment_details"),
    ],
)
def test_stage_to_move(stage, move):
    assert next_move(State(turns=1), stage) == move


def test_payment_ask_after_an_id_asks_for_another():
    s = State(turns=4, ids=["a.b@okaxis"])
    assert next_move(s, "payment_ask") == "confirm_and_ask_alt"
    assert next_move(s, "threat") == "scared_comply"


def test_wind_down_after_two_ids_or_too_many_turns():
    assert next_move(State(turns=5, ids=["a@okaxis", "b@ybl"]), "payment_ask") == "wind_down"
    assert next_move(State(turns=MAX_TURNS), "hook") == "wind_down"
    assert next_move(State(turns=MAX_TURNS - 1), "hook") == "confused_engage"


def test_kamla_is_never_faster_than_her_typing():
    assert wasted_seconds("x" * 40, "y" * 30, 1.0) == pytest.approx(10 + 20)
    assert wasted_seconds("x" * 40, "y" * 30, 45.0) == pytest.approx(10 + 45)


SCENARIO = {
    "collect": [{"kind": "upi", "value": "a.b@okaxis"}],
    "steps": [
        {"stage": "hook", "lines": ["hello one", "hello two"]},
        {"stage": "payment_ask", "reveal": 0, "lines": ["pay {upi}"]},
    ],
}


def test_simulator_advances_on_a_question():
    sim = Simulator(SCENARIO, seed=1)
    assert sim.opening().startswith("hello")
    assert sim.respond("Who is this beta?") == "pay a.b@okaxis"
    assert sim.respond("Which id, say it again?") is None


def test_simulator_repeats_then_pushes_on():
    sim = Simulator(SCENARIO, seed=1)
    first = sim.opening()
    second = sim.respond("Haan ji.")
    assert second.startswith("hello") and second != first
    sim.respond("Okay.")
    assert sim.respond("Hmm.") == "pay a.b@okaxis"


def test_asks_something():
    assert asks_something("Which UPI should I type")
    assert asks_something("ok?")
    assert not asks_something("Haan ji, theek hai.")
