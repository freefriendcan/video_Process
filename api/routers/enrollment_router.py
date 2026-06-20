from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile

from api.schemas import (
    DeleteEnrollmentResponse,
    EnrolledUserResponse,
    EnrolledUsersResponse,
    EnrollResponse,
    PerImageEnrollment,
)
from services.enrollment_service import (
    EnrollmentImage,
    EnrollmentResult,
    EnrollmentService,
    PerImageEnrollmentResult,
)


def build_router(enrollment_service: EnrollmentService) -> APIRouter:
    router = APIRouter()

    @router.post("/enroll", response_model=EnrollResponse)
    async def enroll(request: Request) -> EnrollResponse:
        label, images = await _parse_enroll_form(request)
        if not 1 <= len(images) <= 7:
            raise HTTPException(status_code=400, detail="enrollment accepts between 1 and 7 images")
        try:
            result = enrollment_service.enroll(label=label, images=images)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = _enroll_response(result)
        if result.embeddings_added <= 0:
            raise HTTPException(status_code=422, detail=response.model_dump(mode="json"))
        return response

    @router.get("/enrolled", response_model=EnrolledUsersResponse)
    async def enrolled() -> EnrolledUsersResponse:
        users = [
            EnrolledUserResponse(
                label=user.label,
                num_embeddings=user.num_embeddings,
                updated_at=user.updated_at,
            )
            for user in enrollment_service.list_enrolled()
        ]
        return EnrolledUsersResponse(users=users)

    @router.delete("/enrolled/{label}", response_model=DeleteEnrollmentResponse)
    async def delete_enrolled(label: str) -> DeleteEnrollmentResponse:
        try:
            result = enrollment_service.delete(label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DeleteEnrollmentResponse(status=result.status, label=result.label)

    return router


async def _parse_enroll_form(request: Request) -> tuple[str, list[EnrollmentImage]]:
    form = await request.form()
    label_value = form.get("label")
    if not isinstance(label_value, str) or not label_value.strip():
        raise HTTPException(status_code=400, detail="label is required")

    images: list[EnrollmentImage] = []
    for key, value in form.multi_items():
        if key == "label":
            continue
        if not isinstance(value, StarletteUploadFile):
            continue
        data = await value.read()
        images.append(
            EnrollmentImage(
                name=value.filename or key,
                field_name=key,
                content_type=value.content_type,
                data=data,
            )
        )

    return label_value.strip(), images


def _enroll_response(result: EnrollmentResult) -> EnrollResponse:
    return EnrollResponse(
        label=result.label,
        embeddings_added=result.embeddings_added,
        per_image=[_per_image_response(item) for item in result.per_image],
        total_embeddings=result.total_embeddings,
    )


def _per_image_response(result: PerImageEnrollmentResult) -> PerImageEnrollment:
    return PerImageEnrollment(
        name=result.name,
        angle=result.angle,
        ok=result.ok,
        reason=result.reason,
    )
