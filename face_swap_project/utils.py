import os
import shutil
import subprocess
from typing import List


def run_cmd(cmd: List[str], workdir: str = None) -> subprocess.CompletedProcess:
    """Run a subprocess command and include stderr in failure messages."""
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=workdir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required executable not found: {cmd[0]}. Install it and ensure it is available in PATH."
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{process.stderr}")
    return process


def ensure_dir(path: str) -> None:
    """Create a directory when it does not exist."""
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    """Remove and recreate a directory so each run starts from a clean state."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to a readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"
