from __future__ import annotations

import asyncio
import json
import threading
import time

from loguru import logger
from websockets.server import WebSocketServerProtocol, serve  # type: ignore[attr-defined]

from config import PipelineConfig
from streaming.vision_state import VisionState


class VisionWSServer:
    """Threaded asyncio WebSocket broadcaster for vision annotations."""

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._host = cfg.ws_vision_host
        self._port = cfg.ws_vision_port
        self._min_interval = 1.0 / cfg.ws_vision_fps if cfg.ws_vision_fps > 0 else 0.0
        self._clients: set[WebSocketServerProtocol] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_future: asyncio.Future | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest_json: str | None = None
        self._last_broadcast_time = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name="vision-ws-server",
        )
        self._thread.start()

        if not self._ready.wait(timeout=5.0):
            logger.warning("Vision WebSocket server did not report ready within 5s")

    def stop(self):
        loop = self._loop
        stop_future = self._stop_future
        if loop and not loop.is_closed() and stop_future and not stop_future.done():
            loop.call_soon_threadsafe(stop_future.set_result, None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Vision WebSocket thread did not exit within 5s")

    def update_state(self, state: VisionState) -> bool:
        payload = json.dumps(state.to_dict(), separators=(",", ":"))
        now = time.monotonic()

        with self._lock:
            self._latest_json = payload
            if now - self._last_broadcast_time < self._min_interval:
                return False
            self._last_broadcast_time = now
            loop = self._loop

        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
            return True
        return False

    def _run_thread(self):
        try:
            asyncio.run(self._run_server())
        except Exception as exc:
            logger.exception("Vision WebSocket server crashed: {}", exc)
            self._ready.set()
        finally:
            self._loop = None
            self._stop_future = None

    async def _run_server(self):
        self._loop = asyncio.get_running_loop()
        self._stop_future = self._loop.create_future()

        async with serve(self._handler, self._host, self._port):
            logger.info("Vision WebSocket active: ws://{}:{}/", self._host, self._port)
            self._ready.set()
            await self._stop_future

        clients = list(self._clients)
        if clients:
            await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        self._clients.clear()
        logger.info("Vision WebSocket stopped")

    async def _handler(self, websocket: WebSocketServerProtocol):
        self._clients.add(websocket)
        try:
            with self._lock:
                latest_json = self._latest_json
            if latest_json:
                await websocket.send(latest_json)
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self, payload: str):
        if not self._clients:
            return

        disconnected = []
        for websocket in list(self._clients):
            try:
                await websocket.send(payload)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self._clients.discard(websocket)
