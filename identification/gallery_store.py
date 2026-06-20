from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

import numpy as np
from loguru import logger

from repositories.face_repository import FaceRepository


class GalleryStore:
    """Lock-guarded in-memory ArcFace gallery backed by ``FaceRepository``."""

    def __init__(self, repository: FaceRepository) -> None:
        self._repository = repository
        self._lock = threading.RLock()
        self._gallery = self._normalized_gallery(repository.get_all_embeddings())

    @property
    def repository(self) -> FaceRepository:
        return self._repository

    def snapshot(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self._gallery)

    def user_count(self) -> int:
        with self._lock:
            return len(self._gallery)

    def total_embeddings(self, label: str) -> int:
        with self._lock:
            embeddings = self._gallery.get(label)
            if embeddings is None:
                return 0
            return int(embeddings.shape[0])

    def upsert(
        self,
        label: str,
        embeddings: Sequence[np.ndarray],
        source_angles: Sequence[str | None],
    ) -> int:
        if not embeddings:
            return self.total_embeddings(label)

        total = self._repository.add_embeddings(label, embeddings, source_angles)
        self.reload()
        logger.info("Enrollment gallery updated: {} now has {} embeddings", label, total)
        return total

    def import_gallery(self, gallery: Mapping[str, np.ndarray]) -> None:
        if not gallery:
            return
        self._repository.import_gallery(gallery)
        self.reload()

    def remove(self, label: str) -> bool:
        removed = self._repository.delete_user(label)
        if removed:
            self.reload()
            logger.info("Enrollment gallery user removed: {}", label)
        return removed

    def reload(self) -> None:
        fresh = self._normalized_gallery(self._repository.get_all_embeddings())
        with self._lock:
            self._gallery = fresh

    @staticmethod
    def _normalized_gallery(gallery: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for label, embeddings in gallery.items():
            array = np.asarray(embeddings, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != 512:
                raise ValueError(
                    f"Gallery entry {label!r} has shape {array.shape}; expected (N, 512)"
                )
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms <= 1e-12] = 1.0
            normalized[label] = (array / norms).astype(np.float32)
        return normalized
