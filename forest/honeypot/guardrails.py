"""Checks around the honeypot: who Kamla may talk to, and what her replies may contain.

A reply that fails is never sent; the agent swaps in a hand-written line and counts it.
"""

import re
from dataclasses import dataclass, field

from forest.intel.extract import extract

MAX_CHARS = 220

# digits with single spaces, dashes, dots or commas between them read as one number
DIGIT_RUN = re.compile(r"\d(?:[\s,.-]?\d){3,}")
EMAIL_OR_UPI = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*")
ROLE_PREFIX = re.compile(
    r"^\s*(?:kamla(?: devi)?|agent|assistant)\s*(?:\([^)]*\))?\s*:\s*", re.IGNORECASE
)
SAYS_PAID = re.compile(
    r"\b(?:i|i've|i have|maine|main)\b[^.?!]{0,25}?\b(?:sent|paid|transferred|deposited|"
    r"bhej diya|bhej diye|kar diya|done the payment)\b"
    r"|\bpayment\b[^.?!]{0,15}\b(?:is |has been )?(?:done|successful|complete|completed|sent)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(
    r"\b(?:not|nahi|nahin|never|can't|cannot|couldn't|haven't|didn't)\b|n't", re.IGNORECASE
)
AI_TALK = re.compile(
    r"\b(?:as an ai|an ai\b|a bot\b|chatbot|language model|ai model|artificial intelligence|"
    r"virtual assistant|i am an assistant|computer program|i am a (?:computer|machine|robot|program)\b)",
    re.IGNORECASE,
)


@dataclass
class Whitelist:
    """What the scammer has sent in this session; Kamla may say these back."""

    upis: set[str] = field(default_factory=set)
    digits: list[str] = field(default_factory=list)

    def add(self, scammer_text: str) -> None:
        for e in extract(scammer_text):
            if e.kind == "upi":
                self.upis.add(e.normalized)
        for m in DIGIT_RUN.finditer(scammer_text):
            self.digits.append(re.sub(r"\D", "", m.group(0)))

    def has_digits(self, digits: str) -> bool:
        return any(digits in d for d in self.digits)


def may_start(sender: str, reported: set[str]) -> bool:
    """Kamla only answers senders that victims have already reported."""
    return sender in reported


def may_reply(history: list[dict]) -> bool:
    """Kamla never initiates: she only speaks after a scammer message."""
    return bool(history) and history[-1]["role"] == "scammer"


def clean(reply: str) -> str:
    """Drop role prefixes, list bullets (the prompt shows examples as a list) and anything
    after a second speaker; trim to whole sentences that fit."""
    text = ROLE_PREFIX.sub("", reply.strip())
    text = re.sub(r"^[-*•]\s+", "", text)
    text = re.split(r"\n|\b(?:Caller|Scammer)\s*:", text)[0].strip().strip('"').strip()
    if len(text) <= MAX_CHARS:
        return text
    cut = max(text.rfind(p, 0, MAX_CHARS) for p in ".?!")
    return text[: cut + 1] if cut > 0 else text


def violations(reply: str, allowed: Whitelist) -> list[str]:
    """Reasons this reply can't be sent; empty when it is fine."""
    found = []
    if not reply:
        found.append("empty")
    if len(reply) > MAX_CHARS:
        found.append("too_long")
    for m in DIGIT_RUN.finditer(reply):
        if not allowed.has_digits(re.sub(r"\D", "", m.group(0))):
            found.append("number")
            break
    upis = {e.normalized for e in extract(reply) if e.kind == "upi"}
    upis |= {m.group(0).lower() for m in EMAIL_OR_UPI.finditer(reply)}
    if upis - allowed.upis:
        found.append("upi_or_email")
    for m in SAYS_PAID.finditer(reply):
        if not NEGATION.search(m.group(0)):
            found.append("claims_payment")
            break
    if AI_TALK.search(reply):
        found.append("ai_reference")
    return found
