"""One honeypot turn: scammer text -> extract -> strategy -> Granite -> guardrails -> reply.

Harvested ids go straight into the intel store, which is also the payment guard's lookup, so the
next payment to one of them is held.
"""

import random
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from forest import security
from forest.honeypot import guardrails, persona
from forest.honeypot.llm import Granite
from forest.honeypot.strategy import State, next_move, wasted_seconds
from forest.intel import classify
from forest.intel.extract import extract
from forest.intel.labels import KINDS
from forest.intel.stage import stage
from forest.intel.store import Store

COLLECT_KINDS = ("upi", "account")  # what counts towards "enough ids" for wind-down
WINDOW = 13  # last 6 exchanges plus the new scammer message
LOW_CONF = 0.6
OTHER_BELOW = 0.4
SUMMARY_CHARS = 500
# One worked example, so the base model copies the shape and sticks to the facts it is given.
SUMMARY_HEAD = (
    "Incident reports for the 1930 cyber crime helpline (India).\n\n"
    "Scam type: lottery\nCaller script: hook, payment ask\nCollection ids: ra****@ybl (upi)\n"
    "Payments held by the bank guard: 1 (INTEL_MATCH)\n"
    "Summary: A caller promising a lottery prize asked the victim to pay a fee to ra****@ybl. "
    "The id is now flagged and 1 payment to it was held by the bank guard.\n\n"
)

# Unmistakable cues that override a low-confidence classifier label. The classifier is weak
# on job_task and family_emergency, which only have small hand-written seeds.
CUE_RULES = [
    ("digital_arrest", r"\b(?:cbi|arrest\w*|warrant|narcotics|drugs|money laundering|cyber cell)\b"),
    ("job_task", r"\b(?:internship|registration fee|joining|task|stipend|part[- ]time|commission)\b"),
    ("family_emergency", r"\b(?:nani|nana|dadi|papa|mummy|accident|hospital|police station|wakil|lawyer)\b"),
    ("kyc_bank", r"\b(?:kyc|otp|cvv|account (?:will be )?(?:block\w*|frozen))\b"),
]  # fmt: skip
CUE_PATTERNS = [(t, re.compile(p, re.IGNORECASE)) for t, p in CUE_RULES]

# The small model sometimes slips into the caller's role.
OFF_CHARACTER = re.compile(
    r"\b(?:madam|ma'am|sir)\b|\bi am (?:a |an |the )?(?:police|officer|inspector|caller|"
    r"bank|hr)\b|how can i help you|\bi am calling\b|\byour (?:account|aadhaar|kyc)\b|"
    r"\bi will give you\b",
    re.IGNORECASE,
)


# The varied prompt repeats the scammer's line just above Kamla's slot, and the small model
# sometimes says it back as her own.
ECHO_WORDS = 5


def echoes_caller(reply: str, caller: str) -> bool:
    """True if the reply repeats a run of ECHO_WORDS or more words from the caller's line."""
    a = re.findall(r"[a-z']+", reply.lower())
    b = re.findall(r"[a-z']+", caller.lower())
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size >= ECHO_WORDS


@dataclass
class Session:
    id: str
    scenario: str | None = None
    reported_type: str | None = None
    history: list[dict] = field(default_factory=list)
    state: State = field(default_factory=State)
    allowed: guardrails.Whitelist = field(default_factory=guardrails.Whitelist)
    harvested: list[dict] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    replies: list[dict] = field(default_factory=list)  # per reply: source, timings, problems
    leaked: int = 0

    def scammer_texts(self) -> list[str]:
        return [h["text"] for h in self.history if h["role"] == "scammer"]


def cue_label(texts: list[str]) -> str | None:
    counts: Counter[str] = Counter()
    for text in texts:
        for scam_type, pat in CUE_PATTERNS:
            counts[scam_type] += len(pat.findall(text))
    best = counts.most_common(1)
    return best[0][0] if best and best[0][1] > 0 else None


def session_label(texts: list[str]) -> dict:
    """Scam type for the whole chat: the classifier, with cue rules taking over when it is
    unsure, and `other` when nothing is confident."""
    try:
        model, prob = classify.classify_conversation(texts)
    except FileNotFoundError:
        model, prob = "other", 0.0
    label, source = model, "model"
    if prob < LOW_CONF:
        cue = cue_label(texts)
        if cue:
            label, source = cue, "rules"
        elif prob < OTHER_BELOW:
            label, source = "other", "low_confidence"
    return {"label": label, "prob": prob, "model_label": model, "source": source}


class Agent:
    def __init__(
        self,
        store: Store,
        granite: Granite | None = None,
        publish: Callable[[str, dict], None] | None = None,
        seed: int | None = None,
        varied: bool = True,
    ) -> None:
        self.store = store
        self.granite = granite
        self.publish = publish or (lambda event, data: None)
        self.rng = random.Random(seed)
        # varied=False keeps the fixed worked-example prompt (the eval's baseline)
        self.varied = varied
        self.seen: set[tuple[str, ...]] = set()

    def start(
        self,
        sender: str,
        reported: set[str],
        scenario: str | None = None,
        reported_type: str | None = None,
    ) -> Session:
        if not guardrails.may_start(sender, reported):
            raise PermissionError(f"{sender} is not a reported number")
        s = Session(id=uuid.uuid4().hex[:12], scenario=scenario, reported_type=reported_type)
        self.store.start_session(s.id, scenario)
        return s

    def scam_type(self, s: Session) -> str:
        """Victims report a number with a scam type; that wins over our guess."""
        return s.reported_type or session_label(s.scammer_texts())["label"]

    def turn(self, s: Session, text: str) -> dict:
        if s.state.ended:
            raise ValueError("session has ended")
        self.store.add_message(s.id, "scammer", text)
        s.history.append({"role": "scammer", "text": text})
        s.allowed.add(text)
        s.state.turns += 1

        entities = extract(text)
        stg = stage(text)
        s.stages.append(stg)
        scam_type = self.scam_type(s)
        new_ids = self._harvest(s, entities, scam_type)
        self.publish(
            "chat",
            {
                "session": s.id,
                "role": "scammer",
                "text": text,
                "entities": [e.value for e in entities if e.kind in KINDS],
            },
        )

        move = next_move(s.state, stg)
        echo = s.state.ids[-1] if move == "confirm_and_ask_alt" else None
        reply, info = self._reply(s, move, echo)

        self.store.add_message(s.id, "agent", reply)
        s.history.append({"role": "agent", "text": reply, "raw": info.pop("raw"), "move": move})
        s.replies.append(info)
        s.state.scammer_seconds += wasted_seconds(text, reply, info["seconds"])
        if move == "wind_down":
            s.state.ended = True
        self.store.update_session(s.id, s.state.turns, s.state.scammer_seconds, s.state.ended)
        self.publish("chat", {"session": s.id, "role": "agent", "text": reply, "entities": []})
        return {
            "reply": reply,
            "extracted": [
                {"kind": e.kind, "value": e.value, "normalized": e.normalized} for e in entities
            ],
            "new_ids": new_ids,
            "stage": stg,
            "move": move,
            "scam_type": scam_type,
            "source": info["source"],
            "problems": info["problems"],
            "seconds": info["seconds"],
            "ended": s.state.ended,
        }

    def _harvest(self, s: Session, entities: list, scam_type: str) -> list[str]:
        seen = {h["normalized"] for h in s.harvested}
        new = []
        for e in entities:
            if e.kind not in KINDS or e.normalized in seen:
                continue
            seen.add(e.normalized)
            row = self.store.add_intel(e.normalized, e.kind, scam_type, s.id, e.confidence)
            s.harvested.append(
                {"kind": e.kind, "normalized": e.normalized, "display": row["display"]}
            )
            if e.kind in COLLECT_KINDS:
                s.state.ids.append(e.normalized)
            self.publish("intel", row)
            new.append(row["display"])
        return new

    def _reply(self, s: Session, move: str, echo: str | None) -> tuple[str, dict]:
        """Granite's line if it passes every check, else a hand-written one."""
        if not guardrails.may_reply(s.history):
            raise RuntimeError("Kamla only replies to a scammer message")
        start = time.perf_counter()
        info: dict = {"move": move, "source": "template", "problems": [], "tokens": 0}
        reply = ""
        if self.granite:
            window = s.history[-WINDOW:]
            if self.granite.chat:
                gen = self.granite.generate(persona.chat_messages(window, move, echo))
                raw, shown = gen.text, []
            else:
                if self.varied:
                    shown = persona.draw_examples(move, self.rng, self.seen)
                    prompt = persona.completion_prompt(window, move, echo, shown)
                else:
                    shown = persona.fixed_examples()
                    prompt = persona.completion_prompt(window, move, echo)
                gen = self.granite.generate(prompt)
                raw = (f" {echo}..." if echo else "") + gen.text
            reply = guardrails.clean(raw)
            info["copied"] = persona.copied(reply, shown) if shown else None
            problems = guardrails.violations(reply, s.allowed)
            info.update(tokens=gen.tokens, tok_per_s=gen.tok_per_s, llm_seconds=gen.seconds)
            info["violations"] = list(problems)
            if gen.timed_out:
                problems.append("timeout")
            if OFF_CHARACTER.search(reply):
                problems.append("off_character")
            if echoes_caller(reply, s.history[-1]["text"]):
                problems.append("echoes_caller")
            # every move but the goodbye has to keep the scammer explaining
            if move != "wind_down" and "?" not in reply and not problems:
                tail = persona.TAILS[move]
                if len(reply) + len(tail) < guardrails.MAX_CHARS:
                    # `raw` stays as generated, so the next prompt still extends the cached one
                    reply = f"{reply.rstrip()} {tail}"
                    info["tail_added"] = True
                else:
                    problems.append("no_question")
            info["problems"] = problems
            if not problems:
                info["source"] = "granite"
        if info["source"] == "template":
            reply = persona.fallback(move, echo, self.rng)
            raw = f" {reply}"
            if guardrails.violations(reply, s.allowed):
                s.leaked += 1
        info["raw"] = raw
        info["seconds"] = time.perf_counter() - start
        return reply, info

    def close(self, s: Session) -> None:
        s.state.ended = True
        self.store.update_session(s.id, s.state.turns, s.state.scammer_seconds, True)

    def report(self, s: Session) -> dict:
        """Incident summary for the 1930 helpline, from the session's structured results."""
        label = session_label(s.scammer_texts())
        hashes = {security.hash_id(h["normalized"]) for h in s.harvested}
        rows = [r for r in self.store.list_intel() if r["id_hash"] in hashes]
        stages = [st for i, st in enumerate(s.stages) if i == 0 or s.stages[i - 1] != st]
        facts = {
            "session": s.id,
            "scam_type": s.reported_type or label["label"],
            "predicted": label,
            "stages": stages,
            "ids": [{"kind": r["kind"], "display": r["display"]} for r in rows],
            "holds": sum(r["hits"] for r in rows),
            "hold_reason": "INTEL_MATCH",
            "turns": s.state.turns,
            "scammer_minutes": round(s.state.scammer_seconds / 60, 1),
        }
        facts["summary"], facts["summary_by"] = self._summary(facts)
        return facts

    def _summary(self, f: dict) -> tuple[str, str]:
        kind = f["scam_type"].replace("_", " ")
        ids = ", ".join(f"{i['display']} ({i['kind']})" for i in f["ids"]) or "none"
        stages = ", ".join(st.replace("_", " ") for st in f["stages"])
        template = (
            f"Reported {kind} scam. The caller went through {stages} and asked for money to "
            f"{len(f['ids'])} collection id(s): {ids}. {f['holds']} payment(s) to these ids were "
            f"held by the bank's guard (INTEL_MATCH)."
        )
        if not self.granite:
            return template, "template"
        facts = (
            f"Scam type: {kind}\nCaller script: {stages}\nCollection ids: {ids}\n"
            f"Payments held by the bank guard: {f['holds']} (INTEL_MATCH)\n"
        )
        allowed = guardrails.Whitelist()
        allowed.add(facts)
        if self.granite.chat:
            prompt = [
                {"role": "system", "content": "You write short incident reports for India's "
                 "1930 cyber crime helpline."},
                {"role": "user", "content": facts + "Write a two-sentence summary. Mention "
                 "every collection id exactly as written."},
            ]  # fmt: skip
        else:
            prompt = SUMMARY_HEAD + facts + "Summary:"
        gen = self.granite.generate(prompt, max_tokens=80, stop=["\n"])
        text = " ".join(gen.text.split())
        text = text[: text.rfind(".") + 1]
        sentences = [x.strip().lower() for x in re.split(r"(?<=\.)\s", text) if x.strip()]
        repeats = len(set(sentences)) < len(sentences)
        bad = set(guardrails.violations(text, allowed)) - {"too_long"}
        # a summary that drops an id the helpline needs is worse than the template
        missing = any(i["display"] not in text for i in f["ids"])
        if gen.timed_out or bad or repeats or missing or not 40 <= len(text) <= SUMMARY_CHARS:
            return template, "template"
        return text, "granite"
