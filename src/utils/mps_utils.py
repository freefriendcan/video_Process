"""Apple Silicon MPS (Metal Performance Shaders) utilities."""

import platform


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (M1/M2/M3).

    Returns:
        True if on Apple Silicon
    """
    return platform.processor() == "arm" and platform.system() == "Darwin"


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        True if MPS is available
    """
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns:
        Dictionary with device info
    """
    info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "apple_silicon": is_apple_silicon(),
    }

    try:
        import torch

        info.update({
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "cuda_available": torch.cuda.is_available(),
        })

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)

    except ImportError:
        info["torch_available"] = False

    return info


def get_optimal_device(prefer_mps: bool = True) -> str:
    """
    Get the optimal device for PyTorch operations.

    Args:
        prefer_mps: Whether to prefer MPS on Apple Silicon

    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    try:
        import torch

        if prefer_mps and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    except ImportError:
        return "cpu"


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()

    print("=== Device Information ===")
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Machine: {info['machine']}")
    print(f"Apple Silicon: {info['apple_silicon']}")

    if "torch_version" in info:
        print(f"PyTorch: {info['torch_version']}")
        print(f"MPS Available: {info['mps_available']}")
        print(f"MPS Built: {info['mps_built']}")
        print(f"CUDA Available: {info['cuda_available']}")

        if info.get("cuda_available"):
            print(f"CUDA Devices: {info['cuda_device_count']}")
            print(f"CUDA Device: {info['cuda_device_name']}")

    print(f"Recommended Device: {get_optimal_device()}")
    print("========================")


def test_mps() -> dict:
    """
    Test MPS functionality with a simple operation.

    Returns:
        Dictionary with test results
    """
    results = {
        "mps_available": False,
        "tensor_creation": False,
        "tensor_operation": False,
        "model_inference": False,
        "error": None,
    }

    try:
        import torch

        if not torch.backends.mps.is_available():
            results["error"] = "MPS not available"
            return results

        results["mps_available"] = True
        device = torch.device("mps")

        # Test tensor creation
        try:
            x = torch.randn(10, 10, device=device)
            results["tensor_creation"] = True
        except Exception as e:
            results["error"] = f"Tensor creation: {e}"
            return results

        # Test tensor operation
        try:
            y = x @ x.T
            results["tensor_operation"] = True
        except Exception as e:
            results["error"] = f"Tensor operation: {e}"
            return results

        # Test model inference
        try:
            from ultralytics import YOLO

            # Create a simple model test
            model = YOLO("yolov8n.pt")
            results["model_inference"] = True
        except Exception as e:
            results["error"] = f"Model inference: {e}"

    except ImportError as e:
        results["error"] = f"Import error: {e}"

    return results
