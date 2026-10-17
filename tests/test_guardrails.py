from forest.honeypot import guardrails as g
from forest.honeypot.persona import FALLBACKS, TAILS, fallback


def allowed_from(*texts: str) -> g.Whitelist:
    w = g.Whitelist()
    for t in texts:
        w.add(t)
    return w


SCAMMER = "Transfer Rs 49,999 to inspector.verify22@okaxis now, or a/c 9000 1111 2222."


def test_session_only_for_reported_sender():
    assert g.may_start("digital_arrest", {"digital_arrest", "kyc_bank"})
    assert not g.may_start("unknown_caller", {"digital_arrest"})


def test_kamla_never_speaks_first():
    assert not g.may_reply([])
    assert not g.may_reply([{"role": "scammer", "text": "hi"}, {"role": "agent", "text": "ji"}])
    assert g.may_reply([{"role": "scammer", "text": "hello madam"}])


def test_echoing_the_scammers_own_upi_is_allowed():
    w = allowed_from(SCAMMER)
    assert g.violations("inspector.verify22@okaxis... is that right beta?", w) == []
    assert g.violations("inspector dot verify22 at the rate okaxis, it failed?", w) == []


def test_any_other_upi_or_email_is_blocked():
    w = allowed_from(SCAMMER)
    assert "upi_or_email" in g.violations("Send it to kamla.devi@oksbi instead?", w)
    assert "upi_or_email" in g.violations("My email is kamla@gmail.com", w)
    assert "upi_or_email" in g.violations("kamla dot devi at the rate ybl", w)


def test_numbers_blocked_unless_scammer_sent_them():
    w = allowed_from(SCAMMER)
    assert "number" in g.violations("My OTP is 4821, beta.", w)
    assert "number" in g.violations("My account is 3021 4455 9001.", w)
    assert "number" in g.violations("Aadhaar 1234-5678-9012", w)
    assert g.violations("Rs 49,999 to 9000 1111 2222, is that right?", w) == []
    assert g.violations("Only 2 minutes beta? Rs 10?", w) == []


def test_never_claims_to_have_paid():
    w = g.Whitelist()
    assert "claims_payment" in g.violations("I have sent the money beta.", w)
    assert "claims_payment" in g.violations("Maine paise bhej diye.", w)
    assert "claims_payment" in g.violations("The payment is done, now what?", w)
    assert g.violations("I have not sent anything yet beta, where should I send it?", w) == []


def test_no_ai_self_reference():
    w = g.Whitelist()
    assert "ai_reference" in g.violations("As an AI language model I cannot do that.", w)
    assert "ai_reference" in g.violations("I am just a chatbot.", w)
    assert "ai_reference" in g.violations("I am a real person, a computer program.", w)
    assert "ai_reference" in g.violations("Beta, I am a machine?", w)
    assert g.violations("Rohan said these phones are very clever now?", w) == []


def test_clean_strips_role_and_trims_length():
    assert g.clean("Kamla (scared, asks how): Hai Ram, what to do?") == "Hai Ram, what to do?"
    assert g.clean(' "Haan ji beta?"\nCaller: pay now') == "Haan ji beta?"
    assert g.clean(" - Hello? Who is this beta?") == "Hello? Who is this beta?"
    long = "Haan ji beta, I am listening. " * 10
    out = g.clean(long)
    assert len(out) <= g.MAX_CHARS and out.endswith(".")
    assert "too_long" in g.violations("a" * 300, g.Whitelist())
    assert "empty" in g.violations("", g.Whitelist())


def test_every_fallback_line_passes():
    w = allowed_from(SCAMMER)
    for move, lines in FALLBACKS.items():
        for _ in range(10):
            assert g.violations(fallback(move, "inspector.verify22@okaxis"), w) == [], move
        for line in lines:
            assert g.violations(line.format(id="that id"), g.Whitelist()) == [], line


def test_question_tails_pass():
    for tail in TAILS.values():
        assert g.violations(tail, g.Whitelist()) == [] and tail.endswith("?")
