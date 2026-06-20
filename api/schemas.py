from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PerImageEnrollment(BaseModel):
    name: str
    angle: str | None = None
    ok: bool
    reason: str | None = None


class EnrollResponse(BaseModel):
    label: str
    embeddings_added: int
    per_image: list[PerImageEnrollment]
    total_embeddings: int


class EnrolledUserResponse(BaseModel):
    label: str
    num_embeddings: int
    updated_at: datetime


class EnrolledUsersResponse(BaseModel):
    users: list[EnrolledUserResponse]


class DeleteEnrollmentResponse(BaseModel):
    status: str
    label: str


class TrackResponse(BaseModel):
    id: int
    user: str
    bbox: tuple[int, int, int, int]


class PersonTrackResponse(TrackResponse):
    reid_ok: bool
    confidence: float


class ActiveTrackingResponse(BaseModel):
    faces: list[TrackResponse]
    persons: list[PersonTrackResponse]
    ts: float


class HealthResponse(BaseModel):
    status: str
    gallery_users: int
    model_loaded: bool
