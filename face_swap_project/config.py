from dataclasses import dataclass
from typing import Optional


@dataclass
class OutputConfig:
    """Video encoding settings for the final render."""

    crf: int = 8
    preset: str = "slower"
    audio_bitrate: str = "192k"
    sharpen: float = 0.2


@dataclass
class EnhanceConfig:
    """Post-processing settings applied to the swapped face region."""

    enabled: bool = True
    strength: float = 0.45
    pad_ratio: float = 0.18


@dataclass
class AppConfig:
    """Runtime configuration for the face swap pipeline."""

    video_path: str
    source_path: str
    out_path: str
    workdir: str
    device: str
    swapper_path: Optional[str]
    extract_fps: Optional[float]
    det_size: int
    codeformer_enabled: bool
    codeformer_repo: Optional[str]
    codeformer_python: str
    codeformer_fidelity: float
    output: OutputConfig
    enhance: EnhanceConfig
