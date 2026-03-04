import cv2
import numpy as np

from face_swap_project.config import EnhanceConfig


def _clip_bbox(image: np.ndarray, bbox: np.ndarray, pad_ratio: float) -> tuple[int, int, int, int]:
    """Expand and clip a face box to image bounds."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def _unsharp_mask(image: np.ndarray, strength: float) -> np.ndarray:
    """Sharpen a face crop while keeping noise growth under control."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _local_contrast(image: np.ndarray) -> np.ndarray:
    """Lift local facial contrast in the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _feather_mask(height: int, width: int) -> np.ndarray:
    """Build a soft mask so enhanced face crops blend back without hard edges."""
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    distance = np.maximum(np.abs(xx), np.abs(yy))
    mask = np.clip((1.0 - distance) / 0.35, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=5)


def enhance_face_region(image: np.ndarray, face, cfg: EnhanceConfig) -> np.ndarray:
    """Apply lightweight post-processing to the swapped face region."""
    if not cfg.enabled:
        return image

    x1, y1, x2, y2 = _clip_bbox(image, face.bbox, cfg.pad_ratio)
    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return image

    enhanced = _local_contrast(roi)
    enhanced = cv2.bilateralFilter(enhanced, d=0, sigmaColor=20, sigmaSpace=10)
    enhanced = _unsharp_mask(enhanced, cfg.strength)

    mask = _feather_mask(enhanced.shape[0], enhanced.shape[1])[..., None]
    blended = roi.astype(np.float32) * (1.0 - mask) + enhanced.astype(np.float32) * mask

    output = image.copy()
    output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return output
