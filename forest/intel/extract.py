"""Pull payment identifiers (UPI ids, bank accounts, phones, links, amounts) out of scam chat."""

import re
from dataclasses import dataclass, field

KNOWN_PSP = {
    "okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "ibl", "axl", "paytm", "apl", "upi",
    "fbl", "kotak", "icici", "sbi", "hdfcbank", "barodampay", "aubank", "jupiteraxis",
    "freecharge",
}  # fmt: skip

UPI = re.compile(
    r"(?<![\w.])([a-zA-Z0-9][a-zA-Z0-9._-]{1,63})@([a-zA-Z][a-zA-Z0-9]{1,30})(?!\w|\.\w|@)"
)
# "ramesh dot k at the rate okaxis" and "ramesh.k @ okaxis"
SPOKEN_UPI = re.compile(
    r"(?<![\w.])([a-zA-Z0-9][a-zA-Z0-9_-]*(?:(?:\s+dot\s+|\.)[a-zA-Z0-9_-]+)*)"
    r"\s+(@|at the rate(?: of)?|at)\s+([a-zA-Z][a-zA-Z0-9]{1,30})\b",
    re.IGNORECASE,
)
ACCOUNT_KEYWORD = re.compile(r"\b(?:a/c|a\.c\.?|acct|acc|account|ac|khata)\b", re.IGNORECASE)
DIGIT_RUN = re.compile(r"(?<!\d)\d(?:[\s-]?\d){8,17}(?!\d)")
IFSC = re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
URL = re.compile(
    r"https?://[^\s<>\"']+"
    r"|(?<![@\w.-])[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.(?:apk|in|com|xyz|top)\b(?![\w@])(?:/[^\s<>\"']*)?"
)
AMOUNT_PREFIX = re.compile(
    r"(?:₹|\brs\.?|\binr|\brupees?)\s*([\d,]+(?:\.\d+)?)(?:\s*(lakh|lac|crore|k)\b)?", re.IGNORECASE
)
AMOUNT_SUFFIX = re.compile(
    r"\b([\d,]+(?:\.\d+)?)(?:\s*(lakh|lac|crore|k)\b)?\s*(?:rupees?|rs)\b", re.IGNORECASE
)
HINDI_NUM = {
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "das": 10, "bees": 20, "pachaas": 50,
    "sau": 100,
}  # fmt: skip
UNIT = {"sau": 100, "hazaar": 1_000, "lakh": 100_000, "lac": 100_000, "crore": 10_000_000}
SUFFIX = UNIT | {"k": 1_000}
HINDI_AMOUNT = re.compile(
    r"\b(?:(?:ek|do|teen|chaar|paanch|das|bees|pachaas|sau)\s+(?:hazaar|lakh|lac|crore|sau)\b\s*)+",
    re.IGNORECASE,
)
HINDI_PAIR = re.compile(r"(\w+)\s+(\w+)")

ACCOUNT_WINDOW = 40
IFSC_WINDOW = 80


@dataclass
class Entity:
    kind: str  # upi | account | phone | url | amount
    value: str  # as written in the message
    normalized: str
    span: tuple[int, int]
    confidence: float
    meta: dict = field(default_factory=dict)  # ifsc for accounts, apk flag for urls


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < b and a < span[1] for a, b in taken)


def _upi(text: str) -> list[Entity]:
    found = []
    for m in UPI.finditer(text):
        psp = m.group(2).lower()
        conf = 0.95 if psp in KNOWN_PSP else 0.7
        found.append(Entity("upi", m.group(0), m.group(0).lower(), m.span(), conf))
    taken = [e.span for e in found]
    for m in SPOKEN_UPI.finditer(text):
        sep, psp = m.group(2).lower(), m.group(3).lower()
        # a bare "at" is common English, so it only counts in front of a real PSP name
        if sep == "at" and psp not in KNOWN_PSP:
            continue
        if _overlaps(m.span(), taken):
            continue
        local = re.sub(r"\s+dot\s+", ".", m.group(1), flags=re.IGNORECASE).lower()
        conf = 0.8 if psp in KNOWN_PSP else 0.6
        found.append(Entity("upi", m.group(0), f"{local}@{psp}", m.span(), conf))
    return found


def _accounts(text: str, ifsc_hits: list[tuple[int, int, str]]) -> list[Entity]:
    found = []
    for m in DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if not 9 <= len(digits) <= 18:
            continue
        if len(digits) == 10 and digits[0] in "6789":
            continue
        near = text[max(0, m.start() - ACCOUNT_WINDOW) : m.end() + ACCOUNT_WINDOW]
        if not ACCOUNT_KEYWORD.search(near):
            continue
        meta, conf = {}, 0.85
        if ifsc_hits:
            a, b, code = min(
                ifsc_hits, key=lambda h: min(abs(h[0] - m.end()), abs(m.start() - h[1]))
            )
            if min(abs(a - m.end()), abs(m.start() - b)) <= IFSC_WINDOW:
                meta["ifsc"], conf = code, 0.95
        found.append(Entity("account", m.group(0), digits, m.span(), conf, meta))
    return found


def _urls(text: str) -> list[Entity]:
    found = []
    for m in URL.finditer(text):
        raw = m.group(0).rstrip(".,;:!?)")
        apk = raw.lower().split("?")[0].endswith(".apk")
        span = (m.start(), m.start() + len(raw))
        found.append(Entity("url", raw, raw.lower(), span, 0.9, {"apk": apk}))
    return found


def _number(digits: str, suffix: str | None) -> float:
    n = float(digits.replace(",", ""))
    return n * SUFFIX[suffix.lower()] if suffix else n


def _amounts(text: str) -> list[Entity]:
    found = []
    for pat in (AMOUNT_PREFIX, AMOUNT_SUFFIX):
        for m in pat.finditer(text):
            try:
                n = _number(m.group(1), m.group(2))
            except ValueError:
                continue
            if _overlaps(m.span(), [e.span for e in found]):
                continue
            found.append(Entity("amount", m.group(0), f"{n:g}", m.span(), 0.9))
    for m in HINDI_AMOUNT.finditer(text):
        total = sum(
            HINDI_NUM[a.lower()] * UNIT[b.lower()] for a, b in HINDI_PAIR.findall(m.group(0))
        )
        span = (m.start(), m.start() + len(m.group(0).rstrip()))
        found.append(Entity("amount", text[span[0] : span[1]], str(total), span, 0.8))
    return found


def extract(text: str) -> list[Entity]:
    """All entities in the message, ordered by position. Spans index into `text`."""
    upis = _upi(text)
    taken = [e.span for e in upis]
    ifsc_hits = [(m.start(), m.end(), m.group(0).upper()) for m in IFSC.finditer(text)]
    accounts = [e for e in _accounts(text, ifsc_hits) if not _overlaps(e.span, taken)]
    taken += [e.span for e in accounts]
    phones = []
    for m in PHONE.finditer(text):
        if _overlaps(m.span(), taken):
            continue
        norm = "+91" + re.sub(r"\D", "", m.group(0))[-10:]
        phones.append(Entity("phone", m.group(0), norm, m.span(), 0.9))
    taken += [e.span for e in phones]
    urls = [e for e in _urls(text) if not _overlaps(e.span, taken)]
    taken += [e.span for e in urls]
    amounts = [e for e in _amounts(text) if not _overlaps(e.span, taken)]
    return sorted(upis + accounts + phones + urls + amounts, key=lambda e: e.span)
