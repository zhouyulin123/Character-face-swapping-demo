import os

import cv2
from tqdm import tqdm

from face_swap_project.codeformer_tools import copy_video, resolve_codeformer_repo, run_codeformer_video
from face_swap_project.config import AppConfig
from face_swap_project.enhancement import enhance_face_region
from face_swap_project.ffmpeg_tools import build_video, extract_audio, extract_frames, mux_audio
from face_swap_project.insightface_runtime import load_models, pick_largest_face, validate_inputs
from face_swap_project.utils import ensure_dir, format_file_size, reset_dir


def get_video_fps(video_path: str) -> float:
    """Read input fps and return a safe fallback when metadata is missing."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return 25.0
    return float(fps)


def get_video_duration(video_path: str) -> float:
    """Estimate video duration from metadata for progress logging."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frame_count and frame_count > 0:
        return float(frame_count / fps)
    return 0.0


def run_pipeline(cfg: AppConfig) -> None:
    """Run the full offline face swap pipeline."""
    validate_inputs(cfg.video_path, cfg.source_path)

    frames_in = os.path.join(cfg.workdir, "frames_in")
    frames_out = os.path.join(cfg.workdir, "frames_out")
    audio_path = os.path.join(cfg.workdir, "audio.m4a")
    swapped_video_path = os.path.join(cfg.workdir, "swapped_no_audio.mp4")
    codeformer_output_dir = os.path.join(cfg.workdir, "codeformer_results")

    ensure_dir(cfg.workdir)
    reset_dir(frames_in)
    reset_dir(frames_out)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    video_size = os.path.getsize(cfg.video_path)
    orig_fps = get_video_fps(cfg.video_path)
    duration = get_video_duration(cfg.video_path)
    estimated_frames = int(round((cfg.extract_fps or orig_fps) * duration)) if duration > 0 else 0

    print(f"[info] Input video size: {format_file_size(video_size)} ({video_size} bytes)")
    if duration > 0:
        print(f"[info] Input duration: {duration:.2f}s, estimated frames: {estimated_frames}")

    print(
        f"[1/4] Extract frames (orig_fps={orig_fps:.3f}, "
        f"extract_fps={'keep' if cfg.extract_fps is None else cfg.extract_fps})"
    )
    extract_frames(cfg.video_path, frames_in, fps=cfg.extract_fps)
    actual_frames = len([name for name in os.listdir(frames_in) if name.lower().endswith(".png")])
    print(f"[1/4] Extracted frames: {actual_frames}")

    print("[1/4] Extract audio")
    try:
        extract_audio(cfg.video_path, audio_path, cfg.output.audio_bitrate)
    except Exception:
        print("  (warn) audio extract failed, output will not contain the original audio")

    print("[2/4] Load InsightFace models")
    app, swapper, providers, swapper_model_path, actual_det_size = load_models(
        cfg.device,
        cfg.det_size,
        cfg.swapper_path,
    )
    print(f"[info] Applied providers: {providers}")
    print(f"[info] Swapper model: {swapper_model_path}")
    print(f"[info] Detection size: {actual_det_size}")

    src_img = cv2.imread(cfg.source_path)
    if src_img is None:
        raise FileNotFoundError(f"Cannot read source image: {cfg.source_path}")

    src_faces = app.get(src_img)
    if not src_faces:
        raise RuntimeError("No face detected in source image. Use a clearer front-face photo.")
    src_face = pick_largest_face(src_faces)

    print("[3/4] Face swap per frame")
    frame_files = sorted(name for name in os.listdir(frames_in) if name.lower().endswith(".png"))
    if not frame_files:
        raise RuntimeError("No frames extracted. Check ffmpeg and the input video.")

    for file_name in tqdm(frame_files):
        in_fp = os.path.join(frames_in, file_name)
        out_fp = os.path.join(frames_out, file_name)

        img = cv2.imread(in_fp)
        if img is None:
            continue

        faces = app.get(img)
        if not faces:
            cv2.imwrite(out_fp, img)
            continue

        tgt_face = pick_largest_face(faces)
        swapped = swapper.get(img, tgt_face, src_face, paste_back=True)
        enhanced = enhance_face_region(swapped, tgt_face, cfg.enhance)
        cv2.imwrite(out_fp, enhanced)

    print("[4/4] Build output video")
    out_fps = cfg.extract_fps if cfg.extract_fps is not None else orig_fps
    if cfg.codeformer_enabled:
        print("[4/4] Build intermediate video for CodeFormer")
        build_video(
            frames_out,
            swapped_video_path,
            fps=out_fps,
            output_cfg=cfg.output,
            audio_path=None,
        )
        repo_path = resolve_codeformer_repo(cfg.codeformer_repo)
        print(f"[4/4] Run CodeFormer from: {repo_path}")
        restored_video = run_codeformer_video(
            video_path=swapped_video_path,
            output_dir=codeformer_output_dir,
            python_executable=cfg.codeformer_python,
            repo_path=repo_path,
            fidelity=cfg.codeformer_fidelity,
        )
        if os.path.exists(audio_path):
            mux_audio(restored_video, audio_path, cfg.out_path, cfg.output.audio_bitrate)
        else:
            copy_video(restored_video, cfg.out_path)
    else:
        build_video(
            frames_out,
            cfg.out_path,
            fps=out_fps,
            output_cfg=cfg.output,
            audio_path=audio_path if os.path.exists(audio_path) else None,
        )
    print(f"Done: {cfg.out_path}")
