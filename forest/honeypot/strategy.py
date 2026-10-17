"""Kamla's next move, chosen by fixed rules from the scammer's script stage. The LLM only
phrases the move it is given."""

from dataclasses import dataclass, field

MAX_TURNS = 14
ENOUGH_IDS = 2

STAGE_MOVES = {
    "hook": "confused_engage",
    "authority_claim": "confused_engage",
    "smalltalk": "confused_engage",
    "threat": "scared_comply",
    "credential_ask": "stall_credential",
}

# Simulated pace for the "scammer minutes wasted" stat: the scammer types at about 4 chars/s and
# Kamla, typing slowly on a phone, at about 1.5 chars/s.
SCAMMER_CPS = 4.0
KAMLA_CPS = 1.5


@dataclass
class State:
    turns: int = 0
    ids: list[str] = field(default_factory=list)  # harvested collection ids, oldest first
    scammer_seconds: float = 0.0
    ended: bool = False


def next_move(state: State, stage: str) -> str:
    """Move for the reply to the scammer message just counted in `state.turns`."""
    if len(state.ids) >= ENOUGH_IDS or state.turns >= MAX_TURNS:
        return "wind_down"
    if stage == "payment_ask":
        return "confirm_and_ask_alt" if state.ids else "ask_payment_details"
    return STAGE_MOVES.get(stage, "confused_engage")


def wasted_seconds(scammer_text: str, reply: str, reply_latency: float) -> float:
    """Scammer time spent on one exchange: typing their message, then waiting for Kamla, who
    is never faster than her typing speed."""
    return len(scammer_text) / SCAMMER_CPS + max(reply_latency, len(reply) / KAMLA_CPS)
