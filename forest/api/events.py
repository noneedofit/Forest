"""In-process event bus feeding the dashboard's SSE stream."""

import asyncio
import json


class EventBus:
    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self) -> asyncio.Queue:
        self.loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def publish(self, event: str, data: dict) -> None:
        """Safe from the event loop or a worker thread (the honeypot runs Granite in one).
        A slow client drops events rather than stalling payments."""
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        if self.loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self._put(msg)
        else:
            self.loop.call_soon_threadsafe(self._put, msg)

    def _put(self, msg: str) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
