"""Script stage of the scammer's latest message, used by the honeypot strategy and dashboard."""

import re

KEYWORDS = {
    "payment_ask": [
        "send", "transfer", "pay", "payment", r"bhej\w*", "jama", "deposit", "upi",
        "scan", "qr", "verification amount", "refundable",
    ],
    "threat": [
        r"arrest\w*", "warrant", "fir", "police", "cbi", "court", r"block\w*", "freeze",
        "frozen", "jail", "gande videos", "legal action",
    ],
    "credential_ask": ["otp", "pin", "password", "cvv", "remote app", "anydesk", "teamviewer"],
    "authority_claim": [
        "inspector", "officer", r"from (?:the )?bank", "customer care", "rbi", "trai", "customs",
        "department", "ministry",
    ],
    "hook": [
        r"congratulations?", "won", "prize", "selected", "job", "internship", "offer",
        "lucky draw",
    ],
}  # fmt: skip
COMPILED = [
    (name, re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE))
    for name, words in KEYWORDS.items()
]


def stage(text: str) -> str:
    """First matching rule wins, in the order above; anything else is small talk."""
    for name, pat in COMPILED:
        if pat.search(text):
            return name
    return "smalltalk"
