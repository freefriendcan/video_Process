"""Base YOLO detector with MPS support for Apple Silicon."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from ..config.settings import Config, get_device


class BaseDetector(ABC):
    """
    Abstract base class for YOLO-based detectors.

    Provides common functionality for model loading, device management,
    and MPS optimization for Apple Silicon.
    """

    def __init__(self, config: Config, model_name: str):
        """
        Initialize the detector.

        Args:
            config: Configuration object
            model_name: Name of the model file (e.g., "yolov10n.pt")
        """
        self.config = config
        self.model_name = model_name
        self.model_path = Path(config.models_dir) / model_name
        self.device = get_device(config)
        self._model: YOLO | None = None

        # Ensure models directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> YOLO:
        """Lazy load the YOLO model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> YOLO:
        """
        Load the YOLO model, downloading if necessary.

        Returns:
            Loaded YOLO model
        """
        # Check if model exists locally
        if not self.model_path.exists():
            print(f"Model {self.model_name} not found locally. Downloading...")

        try:
            # Load model (will download if not found)
            model = YOLO(str(self.model_path) if self.model_path.exists() else self.model_name)

            # Move to appropriate device
            if self.device == "mps" and torch.backends.mps.is_available():
                # MPS doesn't support all operations, but inference generally works
                model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                model.to("cuda")

            return model

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print(f"Falling back to CPU")
            # Fallback to CPU
            model = YOLO(self.model_name if not self.model_path.exists() else str(self.model_path))
            return model

    @abstractmethod
    def detect(self, frame: Any) -> dict:
        """
        Run detection on a frame.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Dictionary containing detection results
        """
        pass

    def get_inference_kwargs(self) -> dict:
        """
        Get keyword arguments for model inference.

        Returns:
            Dictionary of inference parameters
        """
        return {
            "conf": self.config.detection.confidence,
            "iou": self.config.detection.iou,
            "verbose": False,
            "device": self.device,
        }

    def preprocess(self, frame: Any) -> Any:
        """
        Preprocess frame before detection.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Default: no preprocessing
        return frame

    def postprocess(self, results: Any) -> dict:
        """
        Postprocess detection results.

        Args:
            results: Raw detection results from YOLO

        Returns:
            Processed results dictionary
        """
        # Default: return results as-is
        return {"results": results}

    def warmup(self) -> None:
        """Run a warmup inference to initialize model and device."""
        import numpy as np

        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy_frame)

    @classmethod
    def is_mps_available(cls) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        return torch.backends.mps.is_available()

    @classmethod
    def get_device_info(cls) -> dict:
        """Get information about available devices."""
        import torch

        info = {
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)

        return info
