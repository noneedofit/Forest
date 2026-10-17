"""Pseudonymisation, masking and encryption at rest. Keys come from the environment."""

import base64
import hashlib
import hmac
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from forest import config

NONCE_LEN = 12


def _hmac_key(key: str | None) -> bytes:
    key = key or config.HMAC_KEY
    if not key:
        raise RuntimeError("FOREST_HMAC_KEY is not set")
    return key.encode()


def _aes_key(key: str | None) -> bytes:
    raw = base64.b64decode(key or config.ENC_KEY or "")
    if len(raw) != 32:
        raise RuntimeError("FOREST_ENC_KEY must be 32 bytes, base64 encoded")
    return raw


def require_keys() -> None:
    """Fail at startup instead of on the first request."""
    _hmac_key(None)
    _aes_key(None)


def hash_id(raw: str, key: str | None = None) -> str:
    """Keyed hash of an account id or UPI handle; this is all the database ever stores."""
    return hmac.new(_hmac_key(key), raw.strip().lower().encode(), hashlib.sha256).hexdigest()[:16]


def mask_handle(handle: str) -> str:
    """ramesh.k22@okaxis -> ra****@okaxis"""
    name, at, bank = handle.partition("@")
    return name[:2] + "****" + at + bank


def mask_id(raw: str) -> str:
    """Account numbers, phones and URLs: keep the first and last two characters."""
    if len(raw) <= 6:
        return raw[:1] + "****"
    return raw[:2] + "****" + raw[-2:]


def mask(raw: str, kind: str) -> str:
    return mask_handle(raw) if kind == "upi" else mask_id(raw)


def encrypt(text: str, key: str | None = None) -> bytes:
    """AES-256-GCM; the random nonce is the first 12 bytes of the blob."""
    nonce = os.urandom(NONCE_LEN)
    return nonce + AESGCM(_aes_key(key)).encrypt(nonce, text.encode(), None)


def decrypt(blob: bytes, key: str | None = None) -> str:
    try:
        return AESGCM(_aes_key(key)).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], None).decode()
    except InvalidTag as e:
        raise ValueError("ciphertext failed authentication") from e
