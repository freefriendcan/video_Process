from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String
from sqlmodel import Field, Session, SQLModel, create_engine, select


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EnrolledUser(SQLModel, table=True):
    __tablename__ = "enrolled_user"

    id: int | None = Field(default=None, primary_key=True)
    label: str = Field(sa_column=Column(String, unique=True, nullable=False, index=True))
    role: str = Field(default="user", nullable=False)
    created_at: datetime = Field(default_factory=_utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=_utc_now, nullable=False)


class FaceEmbedding(SQLModel, table=True):
    __tablename__ = "face_embedding"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("enrolled_user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    vector: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    dim: int = Field(default=512, nullable=False)
    source_angle: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=_utc_now, nullable=False)


@dataclass(frozen=True)
class EnrolledUserSummary:
    label: str
    num_embeddings: int
    updated_at: datetime


class FaceRepository:
    """SQLite-backed persistence for ArcFace enrollment vectors."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(self._engine)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def upsert_user(self, label: str, role: str = "user") -> EnrolledUser:
        clean_label = _clean_label(label)
        with Session(self._engine) as session:
            user = self._upsert_user_in_session(session, clean_label, role=role)
            session.commit()
            session.refresh(user)
            return user

    def add_embeddings(
        self,
        label: str,
        vectors: Sequence[np.ndarray],
        source_angles: Sequence[str | None],
        role: str = "user",
    ) -> int:
        if len(vectors) != len(source_angles):
            raise ValueError("vectors and source_angles must have the same length")

        clean_label = _clean_label(label)
        with Session(self._engine) as session:
            user = self._upsert_user_in_session(session, clean_label, role=role)
            if user.id is None:
                session.flush()
            if user.id is None:
                raise RuntimeError("Failed to allocate enrolled user id")

            for vector, source_angle in zip(vectors, source_angles, strict=True):
                normalized = _normalize_embedding(vector)
                session.add(
                    FaceEmbedding(
                        user_id=user.id,
                        vector=normalized.astype(np.float32).tobytes(),
                        dim=normalized.shape[0],
                        source_angle=source_angle,
                    )
                )

            user.updated_at = _utc_now()
            session.add(user)
            session.commit()

        return self.count_embeddings(clean_label)

    def import_gallery(
        self,
        gallery: Mapping[str, np.ndarray],
        source_angle: str = "pkl_import",
    ) -> None:
        for label, embeddings in gallery.items():
            array = _embedding_matrix(embeddings)
            angles = [source_angle] * int(array.shape[0])
            self.add_embeddings(label, [row for row in array], angles)

    def list_users(self) -> list[EnrolledUserSummary]:
        summaries: list[EnrolledUserSummary] = []
        with Session(self._engine) as session:
            users = session.exec(select(EnrolledUser).order_by(EnrolledUser.label)).all()
            for user in users:
                if user.id is None:
                    continue
                count = len(
                    session.exec(
                        select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)
                    ).all()
                )
                summaries.append(
                    EnrolledUserSummary(
                        label=user.label,
                        num_embeddings=count,
                        updated_at=user.updated_at,
                    )
                )
        return summaries

    def count_embeddings(self, label: str) -> int:
        clean_label = _clean_label(label)
        with Session(self._engine) as session:
            user = session.exec(
                select(EnrolledUser).where(EnrolledUser.label == clean_label)
            ).first()
            if user is None or user.id is None:
                return 0
            return len(
                session.exec(select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)).all()
            )

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        gallery: dict[str, np.ndarray] = {}
        with Session(self._engine) as session:
            users = session.exec(select(EnrolledUser).order_by(EnrolledUser.label)).all()
            for user in users:
                if user.id is None:
                    continue
                rows = session.exec(
                    select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)
                ).all()
                vectors = [_vector_from_row(row) for row in rows]
                if vectors:
                    gallery[user.label] = np.stack(vectors).astype(np.float32)
        return gallery

    def delete_user(self, label: str) -> bool:
        clean_label = _clean_label(label)
        with Session(self._engine) as session:
            user = session.exec(
                select(EnrolledUser).where(EnrolledUser.label == clean_label)
            ).first()
            if user is None:
                return False
            if user.id is not None:
                rows = session.exec(
                    select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)
                ).all()
                for row in rows:
                    session.delete(row)
            session.delete(user)
            session.commit()
            return True

    @staticmethod
    def _upsert_user_in_session(session: Session, label: str, role: str) -> EnrolledUser:
        user = session.exec(select(EnrolledUser).where(EnrolledUser.label == label)).first()
        if user is not None:
            user.role = role
            user.updated_at = _utc_now()
            session.add(user)
            return user

        user = EnrolledUser(label=label, role=role)
        session.add(user)
        session.flush()
        return user


def _clean_label(label: str) -> str:
    clean = label.strip()
    if not clean:
        raise ValueError("label must not be empty")
    return clean


def _embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 512:
        raise ValueError(f"Expected embedding shape (N, 512), got {array.shape}")
    return np.vstack([_normalize_embedding(row) for row in array]).astype(np.float32)


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != 512:
        raise ValueError(f"Expected 512-d ArcFace embedding, got {array.shape[0]}")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    normalized: np.ndarray = array / norm
    return normalized


def _vector_from_row(row: FaceEmbedding) -> np.ndarray:
    vector = np.frombuffer(row.vector, dtype=np.float32).copy()
    if vector.shape[0] != row.dim:
        raise ValueError(f"Stored embedding dim mismatch: blob={vector.shape[0]} row={row.dim}")
    if row.dim != 512:
        raise ValueError(f"Expected stored ArcFace dim 512, got {row.dim}")
    return _normalize_embedding(vector)
