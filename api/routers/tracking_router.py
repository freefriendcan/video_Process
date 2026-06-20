from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ActiveTrackingResponse, HealthResponse, PersonTrackResponse, TrackResponse
from services.tracking_service import TrackingService


def build_router(tracking_service: TrackingService) -> APIRouter:
    router = APIRouter()

    @router.get("/tracking/active", response_model=ActiveTrackingResponse)
    async def tracking_active() -> ActiveTrackingResponse:
        snapshot = tracking_service.active()
        return ActiveTrackingResponse(
            faces=[
                TrackResponse(id=face.id, user=face.user, bbox=face.bbox)
                for face in snapshot.faces
            ],
            persons=[
                PersonTrackResponse(
                    id=person.id,
                    user=person.user,
                    bbox=person.bbox,
                    reid_ok=person.reid_ok,
                    confidence=person.confidence,
                )
                for person in snapshot.persons
            ],
            ts=snapshot.ts,
        )

    @router.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        health = tracking_service.health()
        return HealthResponse(
            status=health.status,
            gallery_users=health.gallery_users,
            model_loaded=health.model_loaded,
        )

    return router
