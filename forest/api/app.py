"""FastAPI service: scores payments, holds the intel store, streams events to the dashboard."""

import asyncio
import re
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from forest import config, security
from forest.api.events import EventBus
from forest.guard.features import FEATURES
from forest.guard.reasons import TEXT
from forest.guard.score import Guard
from forest.intel.store import Store

WEB = Path(__file__).resolve().parent.parent / "web"
MACHINES = {"9175": "IBM z17", "3931": "IBM z16", "8561": "IBM z15"}
RUPEE = "Rupee"


def read_platform() -> dict:
    """Machine type and the on-chip AI accelerator flag, read once from /proc/cpuinfo."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return {"machine": "unknown", "nnpa": False}
    m = re.search(r"machine\s*=\s*(\d+)", text)
    code = m.group(1) if m else "?"
    nnpa = re.search(r"^features\s*:.*\bnnpa\b", text, re.MULTILINE) is not None
    return {"machine": f"{MACHINES.get(code, 'IBM Z')} ({code})", "nnpa": nnpa}


class Txn(BaseModel):
    ts: str
    src: str
    dst: str
    from_bank: str
    to_bank: str
    amount: float
    currency: str
    fmt: str
    currency_recv: str | None = None
    dst_handle: str | None = None
    label: int | None = None  # dataset truth, logged for evaluation and never given to the model
    victim: bool = False


class IntelIn(BaseModel):
    identifier: str
    kind: str
    scam_type: str | None = None
    source_session: str | None = None
    confidence: float | None = None


class Stats:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.totals = store.decision_totals()
        self.latency: deque[float] = deque(maxlen=10_000)

    def snapshot(self) -> dict:
        self.totals["scammer_minutes"] = self.store.decision_totals()["scammer_minutes"]
        lat = np.array(self.latency) if self.latency else np.zeros(1)
        return {
            **self.totals,
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    security.require_keys()
    app.state.store = Store(config.DB_PATH)
    app.state.guard = Guard(intel=app.state.store)
    # The first zDNN call is slow (tens of ms); take that hit before real payments arrive.
    app.state.guard.model.predict_proba(np.zeros((1, len(FEATURES)), dtype=np.float32))
    app.state.platform = read_platform()
    app.state.bus = EventBus()
    app.state.stats = Stats(app.state.store)
    app.state.granite = None
    yield


app = FastAPI(title="Forest", lifespan=lifespan)


def reason_lines(codes: list[str]) -> list[dict]:
    return [{"code": c, "en": TEXT[c][0], "hi": TEXT[c][1]} for c in codes]


@app.post("/score")
async def score(txn: Txn):
    # No awaits in here, so payments are handled one at a time and the shared feature
    # state can't race.
    guard: Guard = app.state.guard
    store: Store = app.state.store
    try:
        d = guard.score(txn.model_dump(), rollback_stopped=True)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    inr = txn.amount * guard.fx[txn.currency] / guard.fx[RUPEE]
    dst_shown = (
        security.mask_handle(txn.dst_handle) if txn.dst_handle else security.mask_id(txn.dst)
    )
    store.log_decision(
        {
            "ts": txn.ts,
            "src_hash": security.hash_id(txn.src),
            "dst_hash": security.hash_id(txn.dst),
            "dst_display": dst_shown,
            "amount_inr": inr,
            "score": d.score,
            "decision": d.decision,
            "reasons": ",".join(d.reasons),
            "latency_ms": d.latency_ms,
            "label": txn.label,
            "victim": int(txn.victim),
        }
    )
    totals = app.state.stats.totals
    totals["scored"] += 1
    if d.decision == "pause":
        totals["paused"] += 1
    elif d.decision == "hold":
        totals["held"] += 1
        if "INTEL_MATCH" in d.reasons:
            totals["held_by_intel"] += 1
            hit = store.match(dst=txn.dst, handle=txn.dst_handle)
            if hit:
                store.add_hit(hit)
        if txn.label == 1 or (txn.victim and "INTEL_MATCH" in d.reasons):
            totals["inr_protected"] += inr
    app.state.stats.latency.append(d.latency_ms)

    reasons = reason_lines(d.reasons)
    app.state.bus.publish(
        "decision",
        {
            "ts": txn.ts,
            "from": security.mask_id(txn.src),
            "to": dst_shown,
            "amount_inr": inr,
            "score": d.score,
            "decision": d.decision,
            "reasons": reasons,
            "victim": txn.victim,
        },
    )
    if totals["scored"] % 20 == 0 or d.decision == "hold":
        app.state.bus.publish("stats", app.state.stats.snapshot())
    return {
        "decision": d.decision,
        "score": d.score,
        "reasons": reasons,
        "latency_ms": d.latency_ms,
    }


@app.get("/events")
async def events(request: Request):
    bus: EventBus = app.state.bus
    q = bus.subscribe()

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while not await request.is_disconnected():
                try:
                    yield await asyncio.wait_for(q.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    app.state.stats.totals["scammer_minutes"] = app.state.store.decision_totals()["scammer_minutes"]
    return app.state.stats.snapshot()


@app.get("/intel")
async def list_intel():
    return app.state.store.list_intel()


@app.post("/intel")
async def add_intel(body: IntelIn):
    row = app.state.store.add_intel(
        body.identifier, body.kind, body.scam_type, body.source_session, body.confidence
    )
    app.state.bus.publish("intel", row)
    return row


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": app.state.guard is not None,
        "granite_loaded": app.state.granite is not None,
        "tree_format": app.state.guard.tree_format,
        **app.state.platform,
    }


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
