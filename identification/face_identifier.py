from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from loguru import logger

from config import PipelineConfig
from detection.onnx_runtime import BBox, NormalizedKeypoint
from identification.gallery_store import GalleryStore
from repositories.face_repository import FaceRepository
from tracking.tracker_manager import TrackerManager


class _RecognitionModel(Protocol):
    def get_feat(self, img: np.ndarray) -> object:
        ...


class _FaceAlignModule(Protocol):
    def norm_crop(self, img: np.ndarray, landmark: np.ndarray, image_size: int) -> np.ndarray:
        ...


class FaceIdentifier:
    """Local ArcFace identifier backed by the enrolled embeddings gallery."""

    def __init__(
        self,
        cfg: PipelineConfig,
        tracker_mgr: TrackerManager,
        gallery_store: GalleryStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._tracker_mgr = tracker_mgr
        self._gallery_store: GalleryStore | None = gallery_store or self._build_gallery_store(cfg)
        self._gallery: dict[str, np.ndarray] = {}
        self._recognition_model, self._face_align = self._load_arcface_model()

    @classmethod
    def for_enrollment(cls, cfg: PipelineConfig) -> FaceIdentifier:
        """Build an identifier with no gallery, for the enrollment script.

        Loads only the ArcFace model so enrollment uses the exact same
        align+embed path as the live pipeline (guaranteeing the gallery and
        runtime share one embedding space). ``identify`` must not be called.
        """
        self = cls.__new__(cls)
        self._cfg = cfg
        self._gallery = {}
        self._gallery_store = None
        self._recognition_model, self._face_align = self._load_arcface_model()
        return self

    @property
    def gallery_store(self) -> GalleryStore:
        if self._gallery_store is None:
            raise RuntimeError("Enrollment-only FaceIdentifier has no gallery store")
        return self._gallery_store

    @property
    def model_loaded(self) -> bool:
        return self._recognition_model is not None

    def identify(
        self,
        face_roi_raw: np.ndarray,
        keypoints: Sequence[NormalizedKeypoint],
        tracker_id: int,
    ) -> None:
        try:
            if len(keypoints) < 5:
                logger.warning("[{}] ArcFace alignment requires 5 face landmarks", tracker_id)
                self._tracker_mgr.set_user(tracker_id, "Unknown", retry_count=0)
                return

            embedding = self.embed_face(face_roi_raw, keypoints)
            label, score = self._best_gallery_match(embedding)

            if label is not None and score >= self._cfg.face_match_cosine_threshold:
                result = self._tracker_mgr.set_user(tracker_id, label, retry_count=0)
                if result is None:
                    logger.debug(
                        "[{}] Tracker already removed, discarding local ID result",
                        tracker_id,
                    )
                else:
                    logger.info(
                        "Identity verified locally [{}]: {} ({:.3f})",
                        tracker_id,
                        label,
                        score,
                    )
            else:
                self._tracker_mgr.set_user(tracker_id, "Unknown", retry_count=0)
                logger.info("[{}] Unknown face locally (best cosine {:.3f})", tracker_id, score)
        except Exception as exc:
            logger.error("[{}] Local face identification failed: {}", tracker_id, exc)
            self._tracker_mgr.set_user(tracker_id, "Unknown", retry_count=0)

    def embed_face(
        self,
        face_roi_raw: np.ndarray,
        keypoints: Sequence[NormalizedKeypoint],
    ) -> np.ndarray:
        """Align (5 landmarks) + ArcFace embed + L2-normalize.

        Shared by ``identify`` (live) and the enrollment script so both produce
        embeddings in the identical space. ``keypoints`` are normalized to the
        ``face_roi_raw`` crop (use ``crop_keypoints`` to convert from full-frame).
        """
        aligned = self._align_face(face_roi_raw, keypoints)
        return self._embed(aligned)

    @staticmethod
    def crop_keypoints(
        keypoints: Sequence[NormalizedKeypoint],
        face_bbox: BBox,
        frame_size: tuple[int, int],
    ) -> tuple[NormalizedKeypoint, ...]:
        frame_w, frame_h = frame_size
        x, y, w, h = face_bbox
        if w <= 0 or h <= 0:
            return tuple(keypoints)

        cropped: list[NormalizedKeypoint] = []
        for keypoint in keypoints:
            full_x = keypoint.x * frame_w
            full_y = keypoint.y * frame_h
            cropped.append(
                NormalizedKeypoint(
                    x=max(0.0, min(1.0, (full_x - x) / w)),
                    y=max(0.0, min(1.0, (full_y - y) / h)),
                )
            )
        return tuple(cropped)

    def _load_gallery(self, gallery_path: Path) -> dict[str, np.ndarray]:
        if not gallery_path.exists():
            raise FileNotFoundError(f"Enrollment gallery not found: {gallery_path}")

        with gallery_path.open("rb") as handle:
            loaded = pickle.load(handle)

        if not isinstance(loaded, dict):
            raise ValueError("Enrollment gallery must be dict[str, np.ndarray]")

        gallery: dict[str, np.ndarray] = {}
        for label, embeddings in loaded.items():
            if not isinstance(label, str):
                raise ValueError("Enrollment gallery labels must be strings")
            array = np.asarray(embeddings, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != 512:
                raise ValueError(
                    f"Gallery entry {label!r} has shape {array.shape}; expected (N, 512)"
                )
            gallery[label] = self._l2_normalize_rows(array)

        if not gallery:
            raise ValueError("Enrollment gallery is empty")
        return gallery

    def _build_gallery_store(self, cfg: PipelineConfig) -> GalleryStore:
        repository = FaceRepository(cfg.face_db_path)
        store = GalleryStore(repository)
        if store.user_count() > 0:
            return store

        gallery_path = Path(cfg.gallery_path)
        if not gallery_path.exists():
            logger.warning("Enrollment DB is empty and no pickle fallback exists: {}", gallery_path)
            return store

        try:
            fallback_gallery = self._load_gallery(gallery_path)
        except Exception as exc:
            logger.warning("Enrollment pickle fallback could not be loaded: {}", exc)
            return store

        store.import_gallery(fallback_gallery)
        logger.info(
            "Imported pickle enrollment gallery into SQLite: {} users -> {}",
            len(fallback_gallery),
            repository.db_path,
        )
        return store

    def _load_arcface_model(self) -> tuple[_RecognitionModel, _FaceAlignModule]:
        try:
            from insightface.model_zoo import get_model
            from insightface.utils import face_align
            from insightface.utils.storage import ensure_available
        except ImportError as exc:
            raise RuntimeError("insightface is required for local ArcFace identification") from exc

        model_dir = Path(ensure_available("models", self._cfg.arcface_model_name))
        recognition_path = model_dir / "w600k_r50.onnx"
        if not recognition_path.exists():
            raise FileNotFoundError(f"ArcFace recognition model not found: {recognition_path}")

        model_obj: Any = get_model(
            str(recognition_path),
            providers=list(self._cfg.onnx_providers),
        )
        if hasattr(model_obj, "prepare"):
            model_obj.prepare(ctx_id=0)
        if not hasattr(model_obj, "get_feat"):
            raise RuntimeError(f"ArcFace model does not expose get_feat(): {recognition_path}")
        return cast(_RecognitionModel, model_obj), face_align

    def _align_face(
        self,
        face_roi_raw: np.ndarray,
        keypoints: Sequence[NormalizedKeypoint],
    ) -> np.ndarray:
        height, width = face_roi_raw.shape[:2]
        # Keypoints arrive in the YOLO-face native spatial order, which already
        # matches the ArcFace 5-point template row order:
        #   image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
        landmarks = np.array(
            [[point.x * width, point.y * height] for point in keypoints[:5]],
            dtype=np.float32,
        )
        aligned = self._face_align.norm_crop(face_roi_raw, landmark=landmarks, image_size=112)
        aligned_array: np.ndarray = np.asarray(aligned)
        return aligned_array

    def _embed(self, aligned_face: np.ndarray) -> np.ndarray:
        feature = self._recognition_model.get_feat(aligned_face)
        embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != 512:
            raise ValueError(f"ArcFace embedding has dim {embedding.shape[0]}; expected 512")
        return self._l2_normalize(embedding)

    def _best_gallery_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        best_label: str | None = None
        best_score = -1.0

        gallery = (
            self._gallery_store.snapshot()
            if self._gallery_store is not None
            else self._gallery
        )
        for label, gallery_embeddings in gallery.items():
            scores = gallery_embeddings @ embedding
            score = float(np.max(scores))
            if score > best_score:
                best_label = label
                best_score = score

        return best_label, best_score

    @staticmethod
    def _l2_normalize_rows(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        normalized: np.ndarray = array / norms
        return normalized

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        normalized: np.ndarray = vector / norm
        return normalized
