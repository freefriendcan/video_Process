"""Face recognition using DeepFace with tracking, FAN alignment, and incremental learning."""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from deepface import DeepFace
import cv2
from loguru import logger

from ..config.settings import Config
from ..detectors.face import FaceDetector
from .face_aligner import FaceAligner


# Minimum face ROI dimensions for reliable embedding extraction
MIN_FACE_SIZE = 40
# Padding ratio around detected face for alignment context
FACE_PADDING_RATIO = 0.3
# Max embeddings stored per person (cap for incremental learning)
MAX_EMBEDDINGS_PER_PERSON = 20
# Similarity threshold above which incremental learning kicks in
INCREMENTAL_LEARN_THRESHOLD = 0.65
# IoU threshold for face tracking (same face across frames)
TRACKING_IOU_THRESHOLD = 0.4
# Seconds before a tracked face must be re-recognized
TRACKING_TIMEOUT = 2.0

# Quality Gate thresholds for incremental learning
MIN_LEARN_FACE_SIZE = 80          # Minimum face size for learning (vs 40 for recognition)
MIN_LAPLACIAN_VAR = 80            # Blur threshold (lower = blurry)
MAX_POSE_ANGLE = 15.0             # Max yaw/pitch angle in degrees
MIN_TRACK_FRAMES = 3              # Min frames tracked before learning
LEARN_INTERVAL_FRAMES = 30        # Re-evaluate learning every N tracked frames


class TrackedFace:
    """A face being tracked across frames."""

    __slots__ = ("box", "name", "confidence", "last_seen", "frame_count")

    def __init__(self, box: list, name: Optional[str], confidence: float):
        self.box = box
        self.name = name
        self.confidence = confidence
        self.last_seen = time.monotonic()
        self.frame_count = 1

    def update(self, box: list, name: Optional[str] = None, confidence: float = 0.0):
        self.box = box
        self.last_seen = time.monotonic()
        self.frame_count += 1
        if name is not None:
            self.name = name
            self.confidence = confidence

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_seen) > TRACKING_TIMEOUT


class FaceRecognizer:
    """
    Face recognition with FAN alignment, tracking, and incremental learning.

    - FAN-based 68-landmark alignment (replaces legacy Haar Cascade)
    - IoU-based face tracking to skip redundant recognition
    - Incremental learning: auto-adds embeddings after high-confidence matches
    """

    def __init__(self, config: Config):
        self.config = config
        self.face_config = config.face_recognition
        self.threshold = self.face_config.threshold
        self.model_name = self.face_config.model
        self.database_path = Path(self.face_config.database_path)
        self.alignment_backend = self.face_config.alignment_backend

        self.detector = FaceDetector(config, model_name=self.face_config.detector_model)

        # FAN aligner (lazy-loaded, only created if backend is "fan")
        self.aligner: FaceAligner | None = None
        if self.alignment_backend == "fan":
            from ..config.settings import get_device
            self.aligner = FaceAligner(device=get_device(config))
            logger.info("Face alignment backend: FAN (68-landmark)")
        else:
            logger.info(f"Face alignment backend: {self.alignment_backend}")

        # Dict[str, List[np.ndarray]] — multiple embeddings per person
        self.known_faces: Dict[str, List[np.ndarray]] = self._load_database()

        # Face tracking state
        self._tracked_faces: List[TrackedFace] = []

    # ── Database ────────────────────────────────────────────────────────

    def _load_database(self) -> Dict[str, List[np.ndarray]]:
        """Load known faces from pickle file, migrating old format if needed."""
        if self.database_path.exists():
            try:
                with open(self.database_path, "rb") as f:
                    data = pickle.load(f)

                # Migrate old format: Dict[str, ndarray] → Dict[str, List[ndarray]]
                if data and isinstance(next(iter(data.values())), np.ndarray):
                    if next(iter(data.values())).ndim == 1:
                        logger.info("Migrating face database from single to multi-embedding format")
                        data = {name: [emb] for name, emb in data.items()}
                        self.database_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.database_path, "wb") as f:
                            pickle.dump(data, f)

                return data
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
        return {}

    def _save_database(self) -> None:
        """Save known faces to pickle file."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.database_path, "wb") as f:
            pickle.dump(self.known_faces, f)

    # ── Registration ────────────────────────────────────────────────────

    def register_face(
        self,
        face_image: np.ndarray,
        name: str,
        *,
        frame: np.ndarray | None = None,
        box: list | None = None,
    ) -> bool:
        """
        Register a face (appends embedding with alignment).

        When frame + box are provided, a padded ROI is extracted internally
        for embedding consistency with the recognition pipeline.
        Otherwise falls back to using face_image directly (backward compat).

        Args:
            face_image: Cropped face region (used as fallback)
            name: Person's name
            frame: Original full frame (optional, preferred)
            box: Face bounding box [x1, y1, x2, y2] (optional, required with frame)
        """
        try:
            # Prefer frame+box for consistent padded embedding
            if frame is not None and box is not None:
                x1, y1, x2, y2 = box
                face_w, face_h = x2 - x1, y2 - y1
                if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                    logger.warning(f"Face too small ({face_w}x{face_h}), skipping registration")
                    return False
                roi = self._extract_padded_roi(frame, x1, y1, x2, y2)
                logger.debug("Registration: using padded ROI from frame+box")
            else:
                h, w = face_image.shape[:2]
                if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
                    logger.warning(f"Face too small ({w}x{h}), skipping registration")
                    return False
                roi = face_image
                logger.debug("Registration: using provided face_image (no frame+box)")

            embedding = self._extract_embedding(roi)
            if embedding is None:
                return False

            if name not in self.known_faces:
                self.known_faces[name] = []

            self.known_faces[name].append(embedding)
            self._save_database()
            logger.info(f"Registered face for '{name}' (total embeddings: {len(self.known_faces[name])})")
            return True

        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False

    def forget_face(self, name: str) -> bool:
        """Remove a face from the database."""
        if name in self.known_faces:
            del self.known_faces[name]
            self._save_database()
            logger.info(f"Removed face for: {name}")
            return True
        return False

    # ── Incremental Learning ────────────────────────────────────────────

    def _maybe_learn(
        self,
        name: str,
        embedding: np.ndarray,
        similarity: float,
        face_roi: np.ndarray,
        face_w: int,
        face_h: int,
        landmarks_68: np.ndarray | None = None,
        track_frame_count: int = 0,
    ) -> None:
        """Auto-add embedding if confidence is high and quality gate passes."""
        if similarity < INCREMENTAL_LEARN_THRESHOLD:
            logger.debug(f"Learning rejected: low similarity ({similarity:.3f} < {INCREMENTAL_LEARN_THRESHOLD})")
            return

        # Quality gate check
        passed, reason = self._check_quality_gate(
            face_roi, face_w, face_h, landmarks_68, track_frame_count,
        )
        if not passed:
            logger.debug(f"Learning rejected for '{name}': {reason}")
            return

        # Cap check
        embeddings = self.known_faces.get(name, [])
        if len(embeddings) >= MAX_EMBEDDINGS_PER_PERSON:
            logger.debug(f"Learning rejected for '{name}': cap reached ({MAX_EMBEDDINGS_PER_PERSON})")
            return

        # Duplicate check
        for existing in embeddings:
            if self._cosine_similarity(embedding, existing) > 0.95:
                logger.debug(f"Learning rejected for '{name}': duplicate embedding")
                return

        self.known_faces[name].append(embedding)
        self._save_database()
        logger.info(f"Incremental learn: added embedding for '{name}' (total: {len(self.known_faces[name])})")

    def _check_quality_gate(
        self,
        face_roi: np.ndarray,
        face_w: int,
        face_h: int,
        landmarks_68: np.ndarray | None = None,
        track_frame_count: int = 0,
    ) -> tuple[bool, str]:
        """Check if a face meets quality thresholds for incremental learning.

        Returns:
            (passed, reason) - if passed=False, reason explains why
        """
        # 1. Size check
        if face_w < MIN_LEARN_FACE_SIZE or face_h < MIN_LEARN_FACE_SIZE:
            return False, f"face too small ({face_w}x{face_h} < {MIN_LEARN_FACE_SIZE})"

        # 2. Blur check (Laplacian variance)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < MIN_LAPLACIAN_VAR:
            return False, f"too blurry (laplacian={laplacian_var:.1f} < {MIN_LAPLACIAN_VAR})"

        # 3. Pose check (if landmarks available and aligner exists)
        if landmarks_68 is not None and self.aligner is not None:
            yaw, pitch, _roll = self.aligner.estimate_pose(landmarks_68)
            if abs(yaw) > MAX_POSE_ANGLE or abs(pitch) > MAX_POSE_ANGLE:
                return False, f"bad pose (yaw={yaw:.1f}°, pitch={pitch:.1f}° > {MAX_POSE_ANGLE}°)"

        # 4. Temporal consistency check
        if track_frame_count < MIN_TRACK_FRAMES:
            return False, f"not tracked long enough ({track_frame_count} < {MIN_TRACK_FRAMES} frames)"

        return True, "quality gate passed"

    # ── Face Tracking ───────────────────────────────────────────────────

    def _find_tracked_face(self, box: list) -> Optional[TrackedFace]:
        """Find an existing tracked face matching this box by IoU."""
        # Prune expired tracks
        self._tracked_faces = [t for t in self._tracked_faces if not t.is_expired]

        best_track = None
        best_iou = TRACKING_IOU_THRESHOLD

        for track in self._tracked_faces:
            iou = self._compute_iou(box, track.box)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        return best_track

    @staticmethod
    def _compute_iou(box_a: list, box_b: list) -> float:
        """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    # ── Recognition ─────────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> dict:
        """
        Recognize faces in a frame with tracking.

        If a face matches a tracked face (IoU), reuses previous identity
        without re-running embedding extraction. Otherwise runs recognition
        with fast (unaligned) embedding extraction.
        """
        if not self.face_config.enabled or not self.known_faces:
            return self._empty_result()

        detection = self.detector.detect(frame)

        if not detection["faces_found"]:
            return self._empty_result()

        # Get largest face
        face_data = max(
            detection["faces"],
            key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
        )

        x1, y1, x2, y2 = face_data["box"]
        face_w, face_h = x2 - x1, y2 - y1

        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            return self._empty_result()

        box = face_data["box"]

        # Check if this face is already tracked
        tracked = self._find_tracked_face(box)
        if tracked and tracked.name is not None:
            tracked.update(box)
            logger.debug(f"Tracking: reused identity '{tracked.name}' (frame {tracked.frame_count})")

            # Periodic learning: every N frames, re-evaluate quality gate
            if tracked.frame_count % LEARN_INTERVAL_FRAMES == 0:
                face_roi = self._extract_padded_roi(frame, x1, y1, x2, y2)
                embedding = self._extract_embedding(face_roi)
                if embedding is not None:
                    landmarks_68 = None
                    if self.aligner is not None:
                        landmarks_68 = self.aligner.get_landmarks(face_roi)
                    self._maybe_learn(
                        tracked.name, embedding, tracked.confidence,
                        face_roi, face_w, face_h,
                        landmarks_68, tracked.frame_count,
                    )

            return {
                "face_found": True,
                "name": tracked.name,
                "confidence": tracked.confidence,
                "location": box,
                "all_faces": detection["faces"],
            }

        # No track match — run recognition with configured alignment
        face_roi = self._extract_padded_roi(frame, x1, y1, x2, y2)
        embedding = self._extract_embedding(face_roi)
        if embedding is None:
            return self._empty_result()

        best_match = self._find_best_match(embedding)

        if best_match and best_match[1] >= self.threshold:
            name, similarity = best_match
            logger.debug(f"Recognized '{name}' with similarity {similarity:.3f}")

            # Get landmarks for quality gate (if FAN is active)
            landmarks_68 = None
            if self.aligner is not None:
                landmarks_68 = self.aligner.get_landmarks(face_roi)

            frame_count = tracked.frame_count if tracked else 1

            # Incremental learning with quality gate
            self._maybe_learn(
                name, embedding, similarity,
                face_roi, face_w, face_h,
                landmarks_68, frame_count,
            )

            # Update or create track
            if tracked:
                tracked.update(box, name, similarity)
            else:
                self._tracked_faces.append(TrackedFace(box, name, similarity))

            return {
                "face_found": True,
                "name": name,
                "confidence": similarity,
                "location": box,
                "all_faces": detection["faces"],
            }

        # Face detected but not recognized
        sim_score = best_match[1] if best_match else 0.0
        logger.debug(f"Face detected but not recognized (best similarity: {sim_score:.3f})")

        if tracked:
            tracked.update(box)
        else:
            self._tracked_faces.append(TrackedFace(box, None, 0.0))

        return {
            "face_found": True,
            "name": None,
            "confidence": 0.0,
            "location": box,
            "all_faces": detection["faces"],
        }

    def recognize_all(self, frame: np.ndarray) -> list[dict]:
        """Recognize all faces in a frame (no tracking, fast extraction)."""
        results = []

        if not self.face_config.enabled or not self.known_faces:
            return results

        detection = self.detector.detect(frame)

        for face_data in detection["faces"]:
            x1, y1, x2, y2 = face_data["box"]
            face_w, face_h = x2 - x1, y2 - y1

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            face_roi = self._extract_padded_roi(frame, x1, y1, x2, y2)
            embedding = self._extract_embedding(face_roi)

            if embedding is not None:
                best_match = self._find_best_match(embedding)
                name = best_match[0] if best_match and best_match[1] >= self.threshold else None
                confidence = best_match[1] if best_match else 0.0

                if name and best_match:
                    # Get landmarks for quality gate (if FAN is active)
                    landmarks_68 = None
                    if self.aligner is not None:
                        landmarks_68 = self.aligner.get_landmarks(face_roi)

                    # No tracking context — assume temporal consistency met
                    self._maybe_learn(
                        name, embedding, best_match[1],
                        face_roi, face_w, face_h,
                        landmarks_68, MIN_TRACK_FRAMES,
                    )

                results.append({
                    "location": face_data["box"],
                    "name": name,
                    "confidence": confidence,
                })

        return results

    # ── Embedding Extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_padded_roi(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Extract face ROI with padding for alignment context."""
        h, w = frame.shape[:2]
        face_w, face_h = x2 - x1, y2 - y1
        pad_x = int(face_w * FACE_PADDING_RATIO)
        pad_y = int(face_h * FACE_PADDING_RATIO)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        return frame[py1:py2, px1:px2]

    def _extract_embedding(
        self, face_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace.

        Alignment strategy based on config:
          - "fan": FAN 68-landmark → SimilarityTransform → 112x112 → DeepFace (no detection)
          - "deepface": Legacy DeepFace internal alignment (Haar Cascade)
          - "none": Skip alignment, raw crop → DeepFace (no detection)
        """
        try:
            input_image = face_image

            if self.alignment_backend == "fan" and self.aligner is not None:
                # FAN alignment: 68 landmarks → canonical 112x112
                aligned = self.aligner.align(face_image)
                if aligned is not None:
                    input_image = aligned
                    logger.debug("FAN alignment applied")
                else:
                    logger.debug("FAN alignment failed, using raw crop")

                # Always skip DeepFace internal detection AND alignment when using FAN
                result = DeepFace.represent(
                    img_path=input_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                    align=False,  # CRITICAL: prevent double alignment
                )

            elif self.alignment_backend == "deepface":
                # Legacy: let DeepFace handle alignment internally
                result = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                )

            else:
                # "none": no alignment, raw crop
                result = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                    align=False,
                )

            if result and len(result) > 0:
                return np.array(result[0]["embedding"])

        except ValueError:
            # Fallback: try without alignment if detection fails
            try:
                result = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                    align=False,  # Fallback: no alignment
                )
                if result and len(result) > 0:
                    logger.debug("Fallback: extracted without alignment")
                    return np.array(result[0]["embedding"])
            except Exception as e:
                logger.error(f"Fallback embedding extraction failed: {e}")

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")

        return None

    # ── Matching ────────────────────────────────────────────────────────

    def _find_best_match(self, embedding: np.ndarray) -> Optional[tuple[str, float]]:
        """Find best match across all embeddings for all people."""
        if not self.known_faces:
            return None

        best_name = None
        best_similarity = -1.0

        for name, embeddings_list in self.known_faces.items():
            for known_embedding in embeddings_list:
                similarity = self._cosine_similarity(embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name

        return (best_name, best_similarity) if best_name else None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ── Utilities ───────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "face_found": False,
            "name": None,
            "confidence": 0.0,
            "location": None,
            "all_faces": [],
        }

    def list_known_faces(self) -> list[str]:
        return list(self.known_faces.keys())

    def get_face_count(self) -> int:
        return len(self.known_faces)

    def get_embedding_count(self, name: str) -> int:
        return len(self.known_faces.get(name, []))
