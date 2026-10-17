import pytest

from forest.intel.stage import stage


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Please transfer the verification amount to the UPI id now", "payment_ask"),
        ("Ab 5000 rupaye bhejo is number par", "payment_ask"),
        ("You will be arrested, a warrant is ready", "threat"),
        ("Aapka account block ho jayega", "threat"),
        ("Share the OTP that came on your phone", "credential_ask"),
        ("Install AnyDesk so I can help", "credential_ask"),
        ("This is Inspector Sharma speaking, ma'am", "authority_claim"),
        ("Hum customer care se bol rahe hain", "authority_claim"),
        ("Congratulations, you have been selected", "hook"),
        ("Aapne lucky draw jeeta hai", "hook"),
        ("Hello, are you at home?", "smalltalk"),
        ("Namaste ji, kaise hain aap", "smalltalk"),
    ],
)
def test_stage_examples(text, expected):
    assert stage(text) == expected


def test_payment_ask_wins_over_threat():
    assert stage("Pay now or police will come") == "payment_ask"
