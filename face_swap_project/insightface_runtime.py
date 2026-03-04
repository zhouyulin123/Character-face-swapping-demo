import os
from typing import Any, List, Optional, Tuple

import cv2


def validate_inputs(video_path: str, source_path: str) -> None:
    """Check that ffmpeg and the required inputs are available."""
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in PATH. Install ffmpeg or add it to PATH before running.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source image does not exist: {source_path}")

    probe = cv2.VideoCapture(video_path)
    ok = probe.isOpened()
    probe.release()
    if not ok:
        raise RuntimeError(f"Cannot open input video: {video_path}")


def get_execution_config(device: str) -> Tuple[List[str], int]:
    """Resolve ONNX Runtime providers and fail early when CUDA was requested but unavailable."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested, but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


def resolve_swapper_model(user_path: Optional[str] = None) -> str:
    """Locate the InSwapper model from explicit or common cache paths."""
    candidates: List[str] = []
    if user_path:
        candidates.append(user_path)

    home = os.path.expanduser("~")
    candidates.extend(
        [
            "inswapper_128.onnx",
            os.path.join(home, ".insightface", "models", "inswapper_128.onnx"),
            os.path.join(home, ".insightface", "models", "inswapper_128", "inswapper_128.onnx"),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot find inswapper_128.onnx. Download the model manually and pass "
        "--swapper <model_path>, or place it under ~/.insightface/models/."
    )


def pick_largest_face(faces) -> Optional[Any]:
    """Choose the largest detected face as the default swap target."""
    best = None
    best_area = -1
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best = face
            best_area = area
    return best


def load_models(device: str, det_size: int, swapper_path: Optional[str]):
    """Load the InsightFace detector and the swapper model."""
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    providers, ctx_id = get_execution_config(device)
    safe_det_size = normalize_det_size(det_size)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(safe_det_size, safe_det_size))
    swapper_model_path = resolve_swapper_model(swapper_path)
    swapper = get_model(swapper_model_path, providers=providers)
    return app, swapper, providers, swapper_model_path, safe_det_size


def normalize_det_size(det_size: int) -> int:
    """Clamp detector size to a stable multiple of 32 for InsightFace models."""
    if det_size < 320:
        det_size = 320
    remainder = det_size % 32
    if remainder != 0:
        det_size += 32 - remainder
    return det_size
