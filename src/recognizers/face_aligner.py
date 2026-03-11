"""Face alignment using FAN (Face Alignment Network) 68-landmark detection."""

import cv2
import numpy as np
from loguru import logger

# Standard ArcFace/GhostFaceNet canonical face template (112x112)
# Reference points: left_eye, right_eye, nose_tip, left_mouth, right_mouth
ARCFACE_REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Mapping from 68-landmark indices to 5 reference points
# left_eye_center, right_eye_center, nose_tip, left_mouth, right_mouth
LANDMARK_68_TO_5 = {
    "left_eye": list(range(36, 42)),   # 6 points around left eye
    "right_eye": list(range(42, 48)),  # 6 points around right eye
    "nose_tip": [30],                  # nose tip
    "left_mouth": [48],                # left mouth corner
    "right_mouth": [54],               # right mouth corner
}

ALIGNED_FACE_SIZE = (112, 112)


class FaceAligner:
    """
    SOTA face alignment using FAN (Face Alignment Network).

    Detects 68 facial landmarks, extracts 5 reference points,
    computes a SimilarityTransform, and warps the face to a
    canonical 112x112 template compatible with ArcFace/GhostFaceNet.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize FAN model (lazy-loaded on first use).

        Args:
            device: "auto", "cpu", "cuda", or "mps"
        """
        self._fa = None
        self._device = device

    def _ensure_model(self):
        """Lazy-load the FAN model on first use."""
        if self._fa is not None:
            return

        import face_alignment

        device = self._resolve_device()
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
        )
        logger.info(f"FAN face alignment model loaded (device={device})")

    def _resolve_device(self) -> str:
        """Resolve device string for face-alignment library."""
        if self._device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "cpu"  # face-alignment doesn't support MPS, fallback
            except ImportError:
                pass
            return "cpu"

        if self._device == "mps":
            return "cpu"  # face-alignment doesn't support MPS

        return self._device

    def align(self, face_roi: np.ndarray) -> np.ndarray | None:
        """
        Align a face ROI to canonical 112x112 template.

        Args:
            face_roi: Cropped face region (BGR, any size)

        Returns:
            Aligned face (112x112 BGR) or None if no landmarks detected
        """
        self._ensure_model()

        try:
            # FAN expects RGB
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            landmarks = self._fa.get_landmarks(rgb)

            if landmarks is None or len(landmarks) == 0:
                logger.debug("FAN: no landmarks detected in ROI")
                return None

            # Use first face's 68 landmarks
            lm68 = landmarks[0]  # shape (68, 2)

            # Extract 5 reference points from 68
            src_points = self._extract_5_from_68(lm68)

            # Compute and apply similarity transform
            aligned = self._warp_to_canonical(face_roi, src_points)
            return aligned

        except Exception as e:
            logger.error(f"FAN alignment error: {e}")
            return None

    def get_landmarks(self, face_roi: np.ndarray) -> np.ndarray | None:
        """
        Get 68 facial landmarks for a face ROI.

        Args:
            face_roi: Cropped face region (BGR)

        Returns:
            Landmarks array (68, 2) or None
        """
        self._ensure_model()

        try:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            landmarks = self._fa.get_landmarks(rgb)
            if landmarks and len(landmarks) > 0:
                return landmarks[0]
        except Exception as e:
            logger.error(f"FAN landmark error: {e}")

        return None

    @staticmethod
    def _extract_5_from_68(lm68: np.ndarray) -> np.ndarray:
        """
        Extract 5 canonical reference points from 68 landmarks.

        Returns:
            Array of shape (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth
        """
        left_eye = lm68[LANDMARK_68_TO_5["left_eye"]].mean(axis=0)
        right_eye = lm68[LANDMARK_68_TO_5["right_eye"]].mean(axis=0)
        nose_tip = lm68[LANDMARK_68_TO_5["nose_tip"]].mean(axis=0)
        left_mouth = lm68[LANDMARK_68_TO_5["left_mouth"]].mean(axis=0)
        right_mouth = lm68[LANDMARK_68_TO_5["right_mouth"]].mean(axis=0)

        return np.array(
            [left_eye, right_eye, nose_tip, left_mouth, right_mouth],
            dtype=np.float32,
        )

    @staticmethod
    def _warp_to_canonical(
        image: np.ndarray, src_points: np.ndarray
    ) -> np.ndarray:
        """
        Warp face to canonical 112x112 using SimilarityTransform.

        Uses the standard ArcFace reference points as the destination.
        """
        from skimage.transform import SimilarityTransform

        tform = SimilarityTransform()
        tform.estimate(src_points, ARCFACE_REFERENCE_POINTS)

        M = tform.params[:2]  # 2x3 affine matrix
        aligned = cv2.warpAffine(
            image,
            M,
            ALIGNED_FACE_SIZE,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    @staticmethod
    def estimate_pose(landmarks_68: np.ndarray) -> tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll) from 68 facial landmarks.

        Uses cv2.solvePnP with a standard 3D face model.

        Returns:
            (yaw, pitch, roll) in degrees
        """
        # 3D face model: 6 key points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye outer corner
            (225.0, 170.0, -135.0),       # Right eye outer corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0),      # Right mouth corner
        ])

        # Corresponding 2D points from 68 landmarks
        image_points = np.array([
            landmarks_68[30],  # Nose tip
            landmarks_68[8],   # Chin
            landmarks_68[36],  # Left eye outer corner
            landmarks_68[45],  # Right eye outer corner
            landmarks_68[48],  # Left mouth corner
            landmarks_68[54],  # Right mouth corner
        ], dtype=np.float32)

        # Camera intrinsics (approximate for 112x112 aligned face)
        focal_length = 450
        center = (112, 112)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0.0

        return (np.degrees(yaw), np.degrees(pitch), np.degrees(roll))
