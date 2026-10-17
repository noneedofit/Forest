"""SQLite store for scam intel, decisions, honeypot sessions and (encrypted) chat messages.

Identifiers are only ever stored as keyed hashes plus a masked display string.
"""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from forest import security

SCHEMA = """
CREATE TABLE IF NOT EXISTS intel (
  id_hash TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  display TEXT NOT NULL,
  scam_type TEXT,
  source_session TEXT,
  confidence REAL,
  first_seen TEXT NOT NULL,
  hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, ts TEXT, src_hash TEXT, dst_hash TEXT, dst_display TEXT,
  amount_inr REAL, score REAL, decision TEXT, reasons TEXT, latency_ms REAL,
  label INTEGER, victim INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, scenario TEXT, started TEXT, ended TEXT, turns INTEGER,
  scammer_seconds REAL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, session_id TEXT, ts TEXT, role TEXT, body_enc BLOB
);
CREATE TABLE IF NOT EXISTS aliases (
  handle_hash TEXT PRIMARY KEY, handle_display TEXT, dst_account_hash TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    """Also the guard's `IntelLookup`: flagged hashes live in a set so a payment never waits
    on the database."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self.flagged: dict[str, str] = {}
        self._reload_flagged()

    def _reload_flagged(self) -> None:
        with self._lock:
            self.flagged = {}
            for (h,) in self.db.execute("SELECT id_hash FROM intel"):
                self.flagged[h] = h
            rows = self.db.execute(
                "SELECT a.handle_hash, a.dst_account_hash FROM aliases a"
                " JOIN intel i ON i.id_hash = a.handle_hash"
            )
            for handle_hash, account_hash in rows:
                self.flagged[account_hash] = handle_hash

    def add_alias(self, handle: str, dst_account: str) -> None:
        h = security.hash_id(handle)
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO aliases VALUES (?, ?, ?)",
                (h, security.mask_handle(handle), security.hash_id(dst_account)),
            )
            self.db.commit()
            self._reload_flagged()

    def add_intel(
        self,
        identifier: str,
        kind: str,
        scam_type: str | None = None,
        source_session: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        """Insert (or return the existing row for) an identifier. A UPI handle with an alias
        also flags the account the alias points at, which is what payments carry."""
        h = security.hash_id(identifier)
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO intel VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    h,
                    kind,
                    security.mask(identifier, kind),
                    scam_type,
                    source_session,
                    confidence,
                    now(),
                ),
            )
            self.db.commit()
            self.flagged[h] = h
            row = self.db.execute(
                "SELECT dst_account_hash FROM aliases WHERE handle_hash = ?", (h,)
            ).fetchone()
            if row:
                self.flagged[row[0]] = h
            return dict(self.db.execute("SELECT * FROM intel WHERE id_hash = ?", (h,)).fetchone())

    def list_intel(self) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM intel ORDER BY first_seen DESC, rowid DESC")
            return [dict(r) for r in rows]

    def match(self, dst: str | None = None, handle: str | None = None) -> str | None:
        """id_hash of the intel row that flags this payment, if any."""
        for raw in (handle, dst):
            if raw:
                hit = self.flagged.get(security.hash_id(raw))
                if hit:
                    return hit
        return None

    def is_flagged(self, dst: str | None = None, handle: str | None = None) -> bool:
        return self.match(dst, handle) is not None

    def add_hit(self, id_hash: str) -> None:
        with self._lock:
            self.db.execute("UPDATE intel SET hits = hits + 1 WHERE id_hash = ?", (id_hash,))
            self.db.commit()

    def log_decision(self, row: dict) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO decisions (ts, src_hash, dst_hash, dst_display, amount_inr, score,"
                " decision, reasons, latency_ms, label, victim) VALUES (:ts, :src_hash,"
                " :dst_hash, :dst_display, :amount_inr, :score, :decision, :reasons,"
                " :latency_ms, :label, :victim)",
                row,
            )
            self.db.commit()

    def decision_totals(self) -> dict:
        """Counters for /stats, rebuilt from the table after a restart."""
        q = self.db.execute
        with self._lock:
            by = dict(q("SELECT decision, COUNT(*) FROM decisions GROUP BY decision").fetchall())
            intel = q(
                "SELECT COUNT(*) FROM decisions WHERE decision='hold' AND reasons LIKE '%INTEL_MATCH%'"
            ).fetchone()[0]
            protected = q(
                "SELECT COALESCE(SUM(amount_inr), 0) FROM decisions WHERE decision='hold' AND"
                " (label = 1 OR (victim = 1 AND reasons LIKE '%INTEL_MATCH%'))"
            ).fetchone()[0]
            minutes = q("SELECT COALESCE(SUM(scammer_seconds), 0) FROM sessions").fetchone()[0]
        return {
            "scored": sum(by.values()),
            "paused": by.get("pause", 0),
            "held": by.get("hold", 0),
            "held_by_intel": intel,
            "inr_protected": protected,
            "scammer_minutes": minutes / 60,
        }

    def add_message(self, session_id: str, role: str, body: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO messages (session_id, ts, role, body_enc) VALUES (?, ?, ?, ?)",
                (session_id, now(), role, security.encrypt(body)),
            )
            self.db.commit()

    def messages(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT ts, role, body_enc FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"ts": r["ts"], "role": r["role"], "body": security.decrypt(r["body_enc"])}
            for r in rows
        ]
