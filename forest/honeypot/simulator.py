"""Scripted scammer that walks through a scenario's steps.

It moves to the next step only when Kamla's reply asks something or asks for details, the way a
real caller answers a question. A reply that doesn't, gets another line from the same step, and
after two of those the scammer pushes on anyway.
"""

import random
import re
from collections.abc import Callable

from forest.honeypot.scenario import render

ASKS = re.compile(
    r"\?|\b(?:which|what|who|how|where|tell me|say it|spell|repeat|again|slowly|batao|bataiye|"
    r"kaun|kya|kaise|kahan|kidhar|other account|another|some other)\b",
    re.IGNORECASE,
)
MAX_REPEATS = 2


def asks_something(reply: str) -> bool:
    return ASKS.search(reply) is not None


class Simulator:
    def __init__(self, scenario: dict, seed: int | None = None) -> None:
        self.sc = scenario
        self.rng = random.Random(seed)
        self.step = 0
        self.repeats = 0
        self.used: dict[int, set[int]] = {}

    @property
    def done(self) -> bool:
        return self.step >= len(self.sc["steps"])

    def _line(self) -> str:
        step = self.sc["steps"][self.step]
        used = self.used.setdefault(self.step, set())
        fresh = [i for i in range(len(step["lines"])) if i not in used] or list(
            range(len(step["lines"]))
        )
        i = self.rng.choice(fresh)
        used.add(i)
        item = self.sc["collect"][step["reveal"]] if "reveal" in step else None
        return render(step["lines"][i], item)

    def opening(self) -> str:
        return self._line()

    def respond(self, reply: str) -> str | None:
        """The scammer's next message, or None once the script has run out."""
        if asks_something(reply) or self.repeats >= MAX_REPEATS:
            self.step += 1
            self.repeats = 0
        else:
            self.repeats += 1
        return None if self.done else self._line()


def play(
    agent,
    scenario: dict,
    reported: set[str],
    seed: int | None = None,
    on_turn: Callable[[str, dict], None] | None = None,
):
    """Run one scenario against the agent until Kamla winds down or the script runs out.
    Returns the agent's session."""
    s = agent.start(scenario["id"], reported, scenario["id"], scenario["scam_type"])
    sim = Simulator(scenario, seed)
    text = sim.opening()
    while text is not None:
        out = agent.turn(s, text)
        if on_turn:
            on_turn(text, out)
        if out["ended"]:
            break
        text = sim.respond(out["reply"])
    agent.close(s)
    return s
