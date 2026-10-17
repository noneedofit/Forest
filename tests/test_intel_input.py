import pytest
from pydantic import ValidationError

from forest.api.app import IntelIn


def test_known_kind_and_scam_type_pass():
    row = IntelIn(identifier="a.b@ybl", kind="upi", scam_type="digital_arrest")
    assert row.kind == "upi"


def test_scam_type_is_optional():
    assert IntelIn(identifier="9876543210", kind="phone").scam_type is None


@pytest.mark.parametrize("kind", ["email", "UPI", "", "ifsc"])
def test_unknown_kind_is_rejected(kind):
    with pytest.raises(ValidationError):
        IntelIn(identifier="x", kind=kind)


@pytest.mark.parametrize("scam_type", ["phishing", "Lottery", ""])
def test_unknown_scam_type_is_rejected(scam_type):
    with pytest.raises(ValidationError):
        IntelIn(identifier="x", kind="upi", scam_type=scam_type)
