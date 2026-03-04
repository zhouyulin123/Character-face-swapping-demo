import os
from typing import Optional

from face_swap_project.config import OutputConfig
from face_swap_project.utils import ensure_dir, run_cmd


def extract_frames(video_path: str, frames_dir: str, fps: Optional[float] = None) -> None:
    """Decode a video into numbered PNG frames."""
    ensure_dir(frames_dir)
    pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [pattern]
    run_cmd(cmd)


def extract_audio(video_path: str, audio_path: str, audio_bitrate: str) -> None:
    """Transcode the source audio to AAC so final muxing is predictable."""
    aac_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
        audio_path,
    ]
    run_cmd(aac_cmd)


def build_video(
    frames_dir: str,
    out_video_path: str,
    fps: float,
    output_cfg: OutputConfig,
    audio_path: Optional[str] = None,
) -> None:
    """Encode processed frames into the final mp4 and mux audio when available."""
    pattern = os.path.join(frames_dir, "frame_%06d.png")
    video_filter = []
    if output_cfg.sharpen > 0:
        video_filter = ["-vf", f"unsharp=5:5:{output_cfg.sharpen}:5:5:0.0"]
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
    ]
    if audio_path and os.path.exists(audio_path):
        cmd += ["-i", audio_path]
    cmd += video_filter
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        str(output_cfg.crf),
        "-preset",
        output_cfg.preset,
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio_path and os.path.exists(audio_path):
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", output_cfg.audio_bitrate, "-shortest"]
    cmd += [out_video_path]
    run_cmd(cmd)


def mux_audio(video_path: str, audio_path: str, out_video_path: str, audio_bitrate: str) -> None:
    """Mux a prepared audio track into an already rendered video."""
    ensure_dir(os.path.dirname(out_video_path) or ".")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        out_video_path,
    ]
    run_cmd(cmd)
