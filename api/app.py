from __future__ import annotations

from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.enrollment_router import build_router as build_enrollment_router
from api.routers.tracking_router import build_router as build_tracking_router
from services.enrollment_service import EnrollmentService
from services.tracking_service import TrackingService


def create_app(
    enrollment_service: EnrollmentService,
    tracking_service: TrackingService,
    cors_origins: Sequence[str] = ("*",),
) -> FastAPI:
    app = FastAPI(title="video_process vision API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_enrollment_router(enrollment_service))
    app.include_router(build_tracking_router(tracking_service))
    return app
