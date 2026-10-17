import pytest

from forest.intel import classify as c

TOY = [
    ("your kyc expires today share otp to update", "kyc_bank"),
    ("bank account blocked update kyc now", "kyc_bank"),
    ("kyc pending click link update account", "kyc_bank"),
    ("you won the lucky draw prize pay processing fee", "lottery"),
    ("lottery winner claim your prize money today", "lottery"),
    ("congratulations prize draw winner claim reward", "lottery"),
    ("lunch at one tomorrow with the family", "legit"),
    ("see you at the temple on sunday evening", "legit"),
    ("dinner is ready come home soon", "legit"),
]


@pytest.fixture
def toy_model(monkeypatch):
    model = c.fit(TOY)
    c._model.cache_clear()
    monkeypatch.setattr(c, "_model", lambda: model)
    return model


def test_normalise_drops_digits_and_punctuation():
    assert c.normalise("Pay Rs. 4,999 NOW!!") == "pay rs now"


def test_numbers_do_not_make_a_new_row():
    rows = [("send Rs 500 to a/c 1234", "x"), ("Send Rs 900 to a/c 9876", "x")]
    kept, conflicts = c.dedupe(rows)
    assert len(kept) == 1
    assert conflicts == 0


def test_conflicting_labels_are_dropped():
    rows = [("hello ji", "legit"), ("Hello ji!", "family_emergency"), ("other text", "legit")]
    kept, conflicts = c.dedupe(rows)
    assert kept == [("other text", "legit")]
    assert conflicts == 1


def test_group_key_ignores_the_tail_of_a_template():
    a = "Your SBI account 1234 is blocked today, call 98765 now"
    b = "your sbi account 9999 is blocked today, visit the branch"
    assert c.group_key(a) == c.group_key(b)


def test_classify_returns_label_and_probability(toy_model):
    label, p = c.classify("update your kyc otp account blocked")
    assert label == "kyc_bank"
    assert 0.3 < p <= 1.0


def test_conversation_label_sums_probabilities(toy_model):
    msgs = ["lunch at one tomorrow", "you won the lucky draw prize", "claim your prize reward"]
    label, share = c.classify_conversation(msgs)
    assert label == "lottery"
    assert 0 < share < 1
