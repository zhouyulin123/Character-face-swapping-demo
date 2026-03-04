import argparse

from face_swap_project.config import AppConfig, EnhanceConfig, OutputConfig
from face_swap_project.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the project."""
    parser = argparse.ArgumentParser(description="Offline video face swap demo with optional face enhancement.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--source", required=True, help="Source face image path.")
    parser.add_argument("--out", required=True, help="Output video path.")
    parser.add_argument("--workdir", default="workdir_swap", help="Temporary working directory.")
    parser.add_argument("--swapper", default=None, help="Path to inswapper_128.onnx.")
    parser.add_argument("--fps", type=float, default=None, help="Optional extraction fps override.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument("--det-size", type=int, default=1024, help="Face detection size. Larger is slower but steadier.")
    parser.add_argument("--use-codeformer", action="store_true", help="Run official CodeFormer on the swapped video.")
    parser.add_argument("--codeformer-repo", default=None, help="Path to a local official CodeFormer repository.")
    parser.add_argument(
        "--codeformer-python",
        default="python",
        help="Python executable used to run CodeFormer inference_codeformer.py.",
    )
    parser.add_argument(
        "--codeformer-fidelity",
        type=float,
        default=0.7,
        help="CodeFormer fidelity weight. Lower tends to restore more aggressively.",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable post-swap face enhancement.")
    parser.add_argument(
        "--enhance-strength",
        type=float,
        default=0.45,
        help="Sharpening strength for post-swap face enhancement.",
    )
    parser.add_argument("--crf", type=int, default=16, help="Output video quality for libx264. Lower means higher quality.")
    parser.add_argument(
        "--preset",
        default="slower",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="x264 encoding preset.",
    )
    parser.add_argument("--audio-bitrate", default="192k", help="Audio bitrate for muxed output.")
    parser.add_argument(
        "--final-sharpen",
        type=float,
        default=0.2,
        help="Mild whole-frame sharpen strength during final ffmpeg encoding. Set 0 to disable.",
    )
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    """Convert parsed CLI arguments into the pipeline config object."""
    return AppConfig(
        video_path=args.video,
        source_path=args.source,
        out_path=args.out,
        workdir=args.workdir,
        device=args.device,
        swapper_path=args.swapper,
        extract_fps=args.fps,
        det_size=args.det_size,
        codeformer_enabled=args.use_codeformer,
        codeformer_repo=args.codeformer_repo,
        codeformer_python=args.codeformer_python,
        codeformer_fidelity=args.codeformer_fidelity,
        output=OutputConfig(
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            sharpen=args.final_sharpen,
        ),
        enhance=EnhanceConfig(enabled=not args.no_enhance, strength=args.enhance_strength),
    )


def main() -> None:
    """Parse arguments and run the face swap project."""
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config(args)
    run_pipeline(cfg)
