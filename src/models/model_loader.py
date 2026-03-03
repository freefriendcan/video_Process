"""Model download and management."""

import hashlib
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

from loguru import logger


class ModelLoader:
    """
    Handle downloading and managing YOLO models.

    Models are cached locally in data/models/ directory.
    """

    # Official Ultralytics model URLs
    MODEL_URLS = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt",
        "yolov10n.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
        "yolov10s.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    }

    # YOLOv8-face (from third-party repo)
    FACE_MODEL_URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.1/yolov8n-face.pt"

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model loader.

        Args:
            models_dir: Directory to store models (default: data/models)
        """
        self.models_dir = Path(models_dir) if models_dir else Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """
        Get the local path for a model.

        Args:
            model_name: Name of the model file

        Returns:
            Path to the model (may not exist yet)
        """
        return self.models_dir / model_name

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model if not already present.

        Args:
            model_name: Name of the model to download
            force: Force re-download even if file exists

        Returns:
            Path to the downloaded model
        """
        model_path = self.get_model_path(model_name)

        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path

        # Determine URL
        if model_name in self.MODEL_URLS:
            url = self.MODEL_URLS[model_name]
        elif "face" in model_name:
            url = self.FACE_MODEL_URL
        else:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Downloading {model_name} from {url}")

        try:
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
                if block_num % 10 == 0:
                    logger.info(f"Downloaded {percent:.1f}%")

            urlretrieve(url, model_path, reporthook=report_progress)
            logger.info(f"Successfully downloaded {model_name} to {model_path}")

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise

        return model_path

    def verify_model(self, model_name: str) -> bool:
        """
        Verify a model file exists and is valid.

        Args:
            model_name: Name of the model

        Returns:
            True if model exists and is valid
        """
        model_path = self.get_model_path(model_name)

        if not model_path.exists():
            return False

        # Check file size (models should be at least 1MB)
        if model_path.stat().st_size < 1024 * 1024:
            return False

        return True

    def list_models(self) -> list[dict]:
        """
        List all available models.

        Returns:
            List of model info dictionaries
        """
        models = []

        for model_name in list(self.MODEL_URLS.keys()) + ["yolov8n-face.pt"]:
            model_path = self.get_model_path(model_name)
            info = {
                "name": model_name,
                "exists": model_path.exists(),
                "path": str(model_path),
            }

            if model_path.exists():
                info["size_mb"] = model_path.stat().st_size / (1024 * 1024)

            models.append(info)

        return models

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model file.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if deleted successfully
        """
        model_path = self.get_model_path(model_name)

        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model: {model_name}")
            return True

        return False

    def get_model_size(self, model_name: str) -> Optional[float]:
        """
        Get model file size in MB.

        Args:
            model_name: Name of the model

        Returns:
            Size in MB or None if file doesn't exist
        """
        model_path = self.get_model_path(model_name)

        if model_path.exists():
            return model_path.stat().st_size / (1024 * 1024)

        return None


def list_available_models() -> list[str]:
    """Get list of all available model names."""
    return list(ModelLoader.MODEL_URLS.keys()) + ["yolov8n-face.pt"]
