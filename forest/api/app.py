"""FastAPI service: scores payments, holds the intel store, streams events to the dashboard."""

import asyncio
import logging
import os
import re
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

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
from forest.honeypot import llm
from forest.honeypot.agent import Agent, Session
from forest.honeypot.scenario import load_all
from forest.intel.labels import KINDS, SCAM_TYPES
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
    kind: Literal[*KINDS]
    scam_type: Literal[*SCAM_TYPES] | None = None
    source_session: str | None = None
    confidence: float | None = None


class SessionIn(BaseModel):
    scenario: str


class TurnIn(BaseModel):
    text: str


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
    # Granite's per-reply speed goes to the container log (evidence for the demo)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    security.require_keys()
    app.state.store = Store(config.DB_PATH)
    app.state.guard = Guard(intel=app.state.store)
    # The first zDNN call is slow (tens of ms); take that hit before real payments arrive.
    app.state.guard.model.predict_proba(np.zeros((1, len(FEATURES)), dtype=np.float32))
    app.state.platform = read_platform()
    app.state.bus = EventBus()
    app.state.stats = Stats(app.state.store)
    granite_on = os.environ.get("FOREST_GRANITE", "1") != "0"
    app.state.granite = llm.load() if granite_on else None
    # Both models fit next to the guard: the 2B's weights are an mmap of the file. Measured with
    # replay + a session + a summary running: 3.3 GB peak for this container, 2.4 GB left free.
    summary_on = granite_on and os.environ.get("FOREST_SUMMARY", "1") != "0"
    app.state.summarizer = (
        llm.load(config.SUMMARY_MODEL_PATH, budget=llm.SUMMARY_BUDGET_S) if summary_on else None
    )
    for g in (app.state.granite, app.state.summarizer):
        if g:
            logging.getLogger("forest.granite").info("loaded %s", g.name)
    app.state.agent = Agent(
        app.state.store,
        app.state.granite,
        publish=app.state.bus.publish,
        summarizer=app.state.summarizer,
    )
    app.state.scenarios = load_all()
    app.state.sessions = {}
    app.state.reports = {}
    app.state.tasks = set()
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


def get_session(sid: str) -> Session:
    s = app.state.sessions.get(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    return s


@app.post("/honeypot/sessions")
async def start_session(body: SessionIn):
    """Open a chat for a reported number. Senders are the simulated scenarios here; anything
    not reported gets a 403, because Kamla never starts a conversation herself."""
    scenarios = app.state.scenarios
    reported = {sid for sid, sc in scenarios.items() if sc.get("reported")}
    sc = scenarios.get(body.scenario, {})
    try:
        s = app.state.agent.start(body.scenario, reported, body.scenario, sc.get("scam_type"))
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    app.state.sessions[s.id] = s
    app.state.bus.publish("session", {"session": s.id, "caller": sc.get("caller", "")})
    return {"id": s.id, "scenario": body.scenario, "caller": sc.get("caller")}


@app.post("/honeypot/sessions/{sid}/turn")
async def session_turn(sid: str, body: TurnIn):
    s = get_session(sid)
    agent: Agent = app.state.agent
    try:
        # Granite takes seconds; a worker thread keeps /score answering meanwhile.
        out = await asyncio.to_thread(agent.turn, s, body.text)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    app.state.bus.publish("stats", app.state.stats.snapshot())
    if out["ended"]:
        report = await asyncio.to_thread(agent.report, s, False)
        if agent.summary_model:
            pending = {**report, "summary_by": "pending", "summary_model": agent.summary_model.name}
            app.state.reports[sid] = pending
            app.state.bus.publish("report", pending)
            task = asyncio.create_task(finish_report(sid, report))
            app.state.tasks.add(task)
            task.add_done_callback(app.state.tasks.discard)
        else:
            app.state.reports[sid] = report
            app.state.bus.publish("report", report)
    return out


async def finish_report(sid: str, report: dict) -> None:
    """Granite's summary takes a minute or two with the 2B model; the dashboard shows the
    report as pending until this replaces it."""
    final = await asyncio.to_thread(app.state.agent.write_summary, report)
    app.state.reports[sid] = final
    app.state.bus.publish("report", final)


@app.get("/honeypot/sessions/{sid}/report")
async def session_report(sid: str):
    s = get_session(sid)
    if sid not in app.state.reports or not s.state.ended:
        # facts so far with the template summary; Granite only writes the final one
        return await asyncio.to_thread(app.state.agent.report, s, False)
    return app.state.reports[sid]


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": app.state.guard is not None,
        "granite_loaded": app.state.granite is not None,
        "granite_model": app.state.granite.name if app.state.granite else None,
        "summary_model": app.state.summarizer.name if app.state.summarizer else None,
        "tree_format": app.state.guard.tree_format,
        **app.state.platform,
    }


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
