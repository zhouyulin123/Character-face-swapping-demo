import os
import shutil
from typing import Optional

from face_swap_project.utils import ensure_dir, run_cmd


def resolve_codeformer_repo(user_path: Optional[str] = None) -> str:
    """Locate a local official CodeFormer repository checkout."""
    candidates = []
    if user_path:
        candidates.append(user_path)

    cwd = os.getcwd()
    candidates.extend(
        [
            os.path.join(cwd, "CodeFormer"),
            os.path.join(cwd, "third_party", "CodeFormer"),
            os.path.join(os.path.expanduser("~"), "CodeFormer"),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "inference_codeformer.py")):
            return candidate

    raise FileNotFoundError(
        "CodeFormer repository not found. Clone the official repo from "
        "https://github.com/sczhou/CodeFormer and pass --codeformer-repo <path>."
    )


def run_codeformer_video(
    video_path: str,
    output_dir: str,
    python_executable: str,
    repo_path: str,
    fidelity: float,
) -> str:
    """Run the official CodeFormer video restoration script on a generated video."""
    ensure_dir(output_dir)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    result_video = os.path.join(output_dir, f"{video_stem}.mp4")
    if os.path.exists(result_video):
        os.remove(result_video)

    cmd = [
        python_executable,
        "inference_codeformer.py",
        "-w",
        str(fidelity),
        "--input_path",
        video_path,
        "--output_path",
        output_dir,
    ]
    run_cmd(cmd, workdir=repo_path)
    if not os.path.isfile(result_video):
        raise FileNotFoundError(
            f"CodeFormer finished without producing the expected video: {result_video}"
        )
    return result_video


def copy_video(src_video: str, dst_video: str) -> None:
    """Copy a generated video to the final output path."""
    ensure_dir(os.path.dirname(dst_video) or ".")
    shutil.copy2(src_video, dst_video)
