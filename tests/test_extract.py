import pytest

from forest.intel.extract import extract


def pick(text: str, kind: str):
    return [e for e in extract(text) if e.kind == kind]


def test_plain_upi():
    (e,) = pick("Send it to Inspector.Verify22@okaxis now", "upi")
    assert e.normalized == "inspector.verify22@okaxis"
    assert e.confidence == 0.95


def test_unknown_psp_is_lower_confidence():
    (e,) = pick("pay rohan.k@newbank today", "upi")
    assert e.confidence == 0.7


def test_email_is_not_a_upi():
    assert pick("write to help.desk@gmail.com for the form", "upi") == []


def test_upi_at_end_of_sentence():
    (e,) = pick("Pay to kbc.prize@ybl.", "upi")
    assert e.normalized == "kbc.prize@ybl"


def test_spoken_upi():
    (e,) = pick("my id is cybercell dot verify22 at the rate okaxis", "upi")
    assert e.normalized == "cybercell.verify22@okaxis"
    assert e.confidence == 0.8


def test_spaced_at_sign():
    (e,) = pick("pay ramesh.k @ oksbi please", "upi")
    assert e.normalized == "ramesh.k@oksbi"


def test_bare_at_needs_known_psp():
    assert pick("meet me at the station", "upi") == []
    assert pick("call me at noon", "upi") == []
    (e,) = pick("send to ramesh dot k at okaxis", "upi")
    assert e.normalized == "ramesh.k@okaxis"


def test_account_with_spaces_and_ifsc():
    (e,) = pick("Account number 5010 0234 5678 12, IFSC HDFC0001234", "account")
    assert e.normalized == "50100234567812"
    assert e.meta["ifsc"] == "HDFC0001234"


def test_account_keyword_after_number():
    (e,) = pick("900011112222 is the a/c to use", "account")
    assert e.normalized == "900011112222"


def test_lowercase_ifsc_is_normalised():
    (e,) = pick("a/c 900033334444 ifsc frst0000002", "account")
    assert e.meta["ifsc"] == "FRST0000002"


def test_aadhaar_like_number_is_not_an_account():
    assert pick("your aadhaar 4821 7733 9012 is linked", "account") == []


def test_phone_is_not_an_account():
    text = "a/c holder phone 98765 43210"
    assert pick(text, "account") == []
    (p,) = pick(text, "phone")
    assert p.normalized == "+919876543210"


def test_phone_with_country_code():
    (p,) = pick("call +91 98765-43210 immediately", "phone")
    assert p.normalized == "+919876543210"


def test_short_numbers_are_not_phones():
    assert pick("otp 482913 and case 12345", "phone") == []


def test_apk_link():
    (u,) = pick("install http://kyc-update.xyz/app/sbi_rewards.apk now", "url")
    assert u.meta["apk"] is True


def test_bare_domain_link():
    (u,) = pick("open parcelfee.in/pay to clear it", "url")
    assert u.normalized == "parcelfee.in/pay"
    assert u.meta["apk"] is False


def test_upi_handle_is_not_also_a_url():
    ents = extract("pay kyc.in@ybl")
    assert [e.kind for e in ents] == ["upi"]


def test_email_domain_is_not_a_url():
    assert pick("mail help@bank.com", "url") == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pay Rs 49,999 today", "49999"),
        ("deposit ₹2.5 lakh", "250000"),
        ("send INR 5k", "5000"),
        ("fee of 1500 rupees", "1500"),
        ("do lakh rupees bhejo", "200000"),
        ("paanch hazaar jama karo", "5000"),
        ("ek lakh pachaas hazaar", "150000"),
    ],
)
def test_amounts(text, expected):
    (a,) = pick(text, "amount")
    assert a.normalized == expected


def test_spans_index_into_text():
    text = "pay Rs 500 to a.b@ybl or call 9876543210"
    for e in extract(text):
        assert text[e.span[0] : e.span[1]].strip() == e.value


def test_message_with_everything():
    text = (
        "Madam transfer Rs 49,999 to cybercell.verify22@okaxis, a/c 50100234567812 "
        "IFSC HDFC0001234, helpline +91 98765 43210"
    )
    kinds = sorted(e.kind for e in extract(text))
    assert kinds == ["account", "amount", "phone", "upi"]
