import random

import pytest

from forest.honeypot import guardrails as g
from forest.honeypot import persona as p
from forest.honeypot.agent import OFF_CHARACTER, echoes_caller

HISTORY = [
    {"role": "scammer", "text": "Madam this is the cyber cell."},
    {"role": "agent", "text": "Who is it?", "raw": " Who is it?", "move": "confused_engage"},
    {"role": "scammer", "text": "Pay the fine through UPI now."},
]


@pytest.mark.parametrize("move", list(p.POOLS))
def test_pool_lines_would_pass_every_check(move):
    # a model that copies a pool line must still produce something sendable
    for line in p.POOLS[move]:
        assert g.violations(line, g.Whitelist()) == []
        assert not OFF_CHARACTER.search(line)
        assert move == "wind_down" or "?" in line


def test_examples_never_repeat_an_order():
    rng, seen = random.Random(0), set()
    draws = [tuple(p.draw_examples("stall_credential", rng, seen)) for _ in range(50)]
    assert len(set(draws)) == 50
    assert all(len(d) == p.SHOWN and len(set(d)) == p.SHOWN for d in draws)


def test_varied_prompt_answers_the_last_scammer_line():
    shown = p.POOLS["ask_payment_details"][:3]
    prompt = p.completion_prompt(HISTORY, "ask_payment_details", None, shown)
    tail = prompt[prompt.rindex("- ") :]
    assert '"Pay the fine through UPI now."' in tail
    assert prompt.endswith(f"Kamla ({p.CUES['ask_payment_details']}):")
    assert "Caller: Pay the fine" not in prompt  # stated once, after the examples
    assert all(f"- {line}" in prompt for line in shown)


def test_fixed_prompt_unchanged_without_examples():
    prompt = p.completion_prompt(HISTORY, "ask_payment_details", "x@ybl")
    assert prompt.startswith(p.head())
    assert prompt.endswith("Caller: Pay the fine through UPI now.\n"
                           f"Kamla ({p.CUES['ask_payment_details']}): x@ybl...")  # fmt: skip


def test_copy_detection():
    shown = p.POOLS["confirm_and_ask_alt"][:3]
    assert p.copied(shown[0], shown) == "verbatim"
    # the id replaces "That id" in the live reply; still a copy
    echoed = "upi.id@ybl" + shown[1][len("That id") :]
    assert p.copied(echoed, shown) == "verbatim"
    assert p.copied(shown[0].replace("beta", "ji"), shown) == "near"
    assert p.copied("Haan ji, my neighbour also got such a call. Who are you?", shown) is None
    assert p.copied("", shown) is None


def test_saying_the_callers_line_back_is_caught():
    caller = "Nani, main Rohan bol raha hoon. Phone kharab ho gaya, ye mera naya number hai."
    assert echoes_caller("Nani, main Rohan bol raha hoon. Who is speaking beta?", caller)
    # repeating only the id or a word or two is fine (the confirm move does that on purpose)
    assert not echoes_caller("Rohan? Phone kharab? Who is speaking beta?", caller)
    assert not echoes_caller("court.notice.dept@okaxis... it failed, any other id?",
                             "UPI id hai court.notice.dept@okaxis, jaldi bhejo")  # fmt: skip
    spoken = "ek aur id, narcotics dot clear07 at the rate okicici. Jaldi kariye."
    assert not echoes_caller("Narcotics dot clear07 at the rate okicici? Any other id?", spoken)
