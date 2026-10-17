import base64
import os

import pytest

from forest import security

KEY_A = "key-a"
KEY_B = "key-b"


def enc_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


def test_hash_is_stable_and_keyed():
    h = security.hash_id("ramesh.k22@okaxis", KEY_A)
    assert h == security.hash_id("ramesh.k22@okaxis", KEY_A)
    assert h == security.hash_id(" Ramesh.K22@OKAXIS ", KEY_A)
    assert h != security.hash_id("ramesh.k22@okaxis", KEY_B)
    assert h != security.hash_id("ramesh.k23@okaxis", KEY_A)
    assert len(h) == 16


def test_hash_needs_a_key(monkeypatch):
    monkeypatch.setattr("forest.config.HMAC_KEY", "")
    with pytest.raises(RuntimeError):
        security.hash_id("x")


def test_masking():
    assert security.mask_handle("ramesh.k22@okaxis") == "ra****@okaxis"
    assert security.mask("ramesh.k22@okaxis", "upi") == "ra****@okaxis"
    assert security.mask("8000EBD30", "account") == "80****30"
    assert "ramesh" not in security.mask("ramesh.k22@okaxis", "upi")


def test_encrypt_round_trip():
    key = enc_key()
    blob = security.encrypt("send the OTP, beta", key)
    assert b"OTP" not in blob
    assert security.decrypt(blob, key) == "send the OTP, beta"
    assert security.encrypt("same", key) != security.encrypt("same", key)


def test_tampered_or_wrong_key_fails():
    key = enc_key()
    blob = bytearray(security.encrypt("hello", key))
    blob[-1] ^= 1
    with pytest.raises(ValueError):
        security.decrypt(bytes(blob), key)
    with pytest.raises(ValueError):
        security.decrypt(security.encrypt("hello", key), enc_key())


def test_bad_enc_key_rejected():
    with pytest.raises(RuntimeError):
        security.encrypt("x", base64.b64encode(b"short").decode())
