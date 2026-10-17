"""Reason codes shown to the payer, in English and Hindi."""

# Highest priority first; a decision shows at most three.
PRIORITY = [
    "INTEL_MATCH",
    "MULE_FANIN",
    "FAST_FORWARD",
    "AMOUNT_UNUSUAL",
    "NEW_RECEIVER",
    "NEW_PAYEE",
    "BURST",
]

# Shown when the model alone stops a payment and no rule fires.
FALLBACK = "RISK_PATTERN"

TEXT = {
    "INTEL_MATCH": (
        "This account was reported in a scam conversation.",
        "Yeh khaata ek scam baatcheet mein report hua hai.",
    ),
    "MULE_FANIN": (
        "Many new people have paid this account in the last 24 hours.",
        "Pichhle 24 ghante mein kai naye logon ne is khaate mein paise bheje hain.",
    ),
    "FAST_FORWARD": (
        "This account sends money onward almost as fast as it receives it.",
        "Yeh khaata paise milte hi aage bhej deta hai.",
    ),
    "AMOUNT_UNUSUAL": (
        "This amount is much larger than your usual payments.",
        "Yeh rakam aapke aam bhugtaan se kaafi zyada hai.",
    ),
    "NEW_RECEIVER": (
        "This account is less than a day old.",
        "Yeh khaata ek din se bhi kam purana hai.",
    ),
    "NEW_PAYEE": (
        "You have never paid this account before.",
        "Aapne is khaate mein pehle kabhi paise nahi bheje hain.",
    ),
    "BURST": (
        "You have made several payments in the last hour.",
        "Aapne pichhle ek ghante mein kai bhugtaan kiye hain.",
    ),
    "RISK_PATTERN": (
        "This payment looks like patterns seen in past fraud.",
        "Yeh bhugtaan pehle hue fraud ke pattern jaisa dikhta hai.",
    ),
}


def explain(codes: list[str], lang: str = "en") -> list[str]:
    """Map reason codes to display lines; lang is "en" or "hi"."""
    col = 1 if lang == "hi" else 0
    return [TEXT[c][col] for c in codes]
