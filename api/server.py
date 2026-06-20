from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI
from loguru import logger

from config import PipelineConfig


class VisionAPIServer:
    """Run the FastAPI app in a sibling thread without blocking the camera loop."""

    def __init__(self, cfg: PipelineConfig, app: FastAPI) -> None:
        self._cfg = cfg
        self._app = app
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._cfg.vision_api_enabled:
            logger.info("Vision REST API disabled by config")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        config = uvicorn.Config(
            self._app,
            host=self._cfg.vision_api_host,
            port=self._cfg.vision_api_port,
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="vision-api",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Vision REST API listening on {}:{}",
            self._cfg.vision_api_host,
            self._cfg.vision_api_port,
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._server = None
