"""Granite through llama.cpp (zDNN build in the base image), with a time budget per reply.

Benchmarks on the z17 VM (2 threads, eval/results/granite_bench.json) picked the 350M base f16
model: ~8 tok/s, about 4 s per reply once the prompt is cached. The 1B bf16 decoded at ~1.4 tok/s
(~28 s per reply) with a 6.4 GB peak, which doesn't fit next to the payment guard on an 8 GB VM.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from forest import config

log = logging.getLogger("forest.granite")

N_THREADS = 2  # leaves a core for payment scoring
N_CTX = 2048
MAX_TOKENS = 32
# Warmer sampling and a repeat penalty, so the 350M model copies its examples less often.
TEMPERATURE = 0.85
REPEAT_PENALTY = 1.15
# Measured on the VM: a first turn takes ~13 s (whole prompt), later turns ~4 s because the
# prompt prefix is reused. Past this the reply is dropped for a hand-written line.
BUDGET_S = 20.0


@dataclass
class Generation:
    text: str
    tokens: int
    seconds: float
    tok_per_s: float
    timed_out: bool


class Granite:
    """One loaded model, shared by all sessions; calls are serialised because llama.cpp
    contexts aren't thread-safe."""

    def __init__(
        self,
        path: Path | str | None = None,
        n_threads: int = N_THREADS,
        budget: float = BUDGET_S,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        from llama_cpp import Llama

        self.path = Path(path or config.MODEL_PATH)
        self.name = self.path.name
        # base models have no chat format worth using; they get a completion prompt
        self.chat = "base" not in self.name
        self.budget = budget
        self.max_tokens = max_tokens
        self.llm = Llama(model_path=str(self.path), n_ctx=N_CTX, n_threads=n_threads, verbose=False)
        self.lock = threading.Lock()

    def generate(
        self,
        prompt: str | list[dict],
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = TEMPERATURE,
        repeat_penalty: float = REPEAT_PENALTY,
    ) -> Generation:
        """Stream tokens until a stop string, `max_tokens` or the time budget."""
        opts = {
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or ["\n"],
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "stream": True,
        }
        with self.lock:
            start = time.perf_counter()
            if self.chat:
                stream = self.llm.create_chat_completion(prompt, **opts)
            else:
                stream = self.llm.create_completion(prompt, **opts)
            parts, first, timed_out = [], None, False
            for chunk in stream:
                choice = chunk["choices"][0]
                piece = choice["delta"].get("content") if self.chat else choice["text"]
                if piece:
                    first = first or time.perf_counter()
                    parts.append(piece)
                if time.perf_counter() - start > self.budget:
                    timed_out = True
                    stream.close()
                    break
            end = time.perf_counter()
        n = len(parts)
        decode = end - first if first else 0.0
        gen = Generation(
            text="".join(parts),
            tokens=n,
            seconds=end - start,
            tok_per_s=(n - 1) / decode if n > 1 and decode > 0 else 0.0,
            timed_out=timed_out,
        )
        log.info(
            "%s: %d tokens in %.1f s, decode %.1f tok/s%s",
            self.name, n, gen.seconds, gen.tok_per_s, " (over budget)" if timed_out else "",
        )  # fmt: skip
        return gen


def load(
    path: Path | str | None = None, budget: float = BUDGET_S, max_tokens: int = MAX_TOKENS
) -> Granite | None:
    """The model if its file exists, else None (the agent then uses its hand-written lines)."""
    p = Path(path or config.MODEL_PATH)
    if not p.exists():
        return None
    return Granite(p, budget=budget, max_tokens=max_tokens)
