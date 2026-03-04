# Video Face Swap Project

This folder now contains a small offline face-swap project instead of a single-file demo.

## Files

- `video_marge_demo.py`: Project entry script.
- `face_swap_project/cli.py`: Command-line parsing.
- `face_swap_project/pipeline.py`: Main processing pipeline.
- `face_swap_project/insightface_runtime.py`: InsightFace loading and runtime validation.
- `face_swap_project/enhancement.py`: Post-swap face enhancement logic.
- `face_swap_project/ffmpeg_tools.py`: Frame extraction, audio extraction, and final muxing.
- `face_swap_project/config.py`: Config objects.
- `requirements.txt`: Python dependencies for the demo.

## Environment

- Python 3.9
- Windows x64
- FFmpeg available in `PATH`
- NVIDIA GPU is optional, but `onnxruntime-gpu` is recommended for faster inference

Install Python dependencies:

```powershell
D:\python3\python.exe -m pip install -r .\requirements.txt
```

Verify the runtime:

```powershell
D:\python3\python.exe -c "import insightface, onnxruntime as ort; print(insightface.__version__); print(ort.get_available_providers())"
ffmpeg -version
```

## Required model files

The project currently needs these models:

- `buffalo_l`
- `inswapper_128.onnx`

Notes:

- `buffalo_l` is usually downloaded automatically by InsightFace on first run
- `inswapper_128.onnx` should be prepared locally in advance

`inswapper_128.onnx` can be placed in any of these locations:

- `.\inswapper_128.onnx`
- `C:\Users\HP\.insightface\models\inswapper_128.onnx`
- `C:\Users\HP\.insightface\models\inswapper_128\inswapper_128.onnx`

You can also pass it explicitly with `--swapper`.

## Current enhancement strategy

The project now adds a post-swap enhancement step:

1. Run `inswapper_128.onnx`
2. Apply local contrast enhancement, bilateral filtering, and sharpening on the swapped face region
3. Blend the enhanced face crop back into the full frame with a soft mask

This is a lightweight enhancement path. It improves softness and weak facial detail without adding a heavy PyTorch-based restoration dependency.

## About stronger models

If the question is whether there is something stronger than `inswapper_128`, the practical answer is nuanced:

- In the current InsightFace-based path, `inswapper_128` is still the most common and easiest swapper to deploy.
- In practice, bigger quality gains usually come from adding a face restoration stage after swapping, such as `CodeFormer` or `GFPGAN`.
- Those are restoration pipelines rather than direct drop-in official InsightFace swapper replacements for this project.

That is why this project now improves quality in two ways first: higher-quality encoding and a built-in post-swap face enhancement step.

## CodeFormer integration

The project now supports the official `CodeFormer` repository as an optional second-stage restoration step.

First prepare a local checkout of the official repository:

```powershell
git clone https://github.com/sczhou/CodeFormer.git
```

Then install its own dependencies and weights as described by the official project.

Example usage:

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\video_datas\minvideo1.mp4" --source ".\photo\OIP.webp" --out ".\video_result\result_codeformer.mp4" --device cuda --use-codeformer --codeformer-repo "D:\path\to\CodeFormer" --codeformer-python D:\python3\python.exe
```

Related arguments:

- `--use-codeformer`: enable the second-stage CodeFormer restoration pass
- `--codeformer-repo`: local path to the official CodeFormer repository
- `--codeformer-python`: Python executable used to launch CodeFormer
- `--codeformer-fidelity`: CodeFormer fidelity weight, default `0.7`

## Usage

Run from this folder:

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result1.mp4" --device cuda
```

If you want to pass the swapper model path explicitly:

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result1.mp4" --device cuda --swapper "C:\Users\HP\.insightface\models\inswapper_128\inswapper_128.onnx"
```

For higher-quality output:

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result_hq.mp4" --device cuda --det-size 1024 --crf 14 --preset slow
```

## Arguments

- `--video`: Input video path
- `--source`: Source face image path
- `--out`: Output video path
- `--workdir`: Temporary working directory, default is `workdir_swap`
- `--fps`: Optional extraction fps override; by default the script keeps the source fps
- `--device`: `cpu` or `cuda`
- `--det-size`: Detection size; larger is steadier but slower
- `--no-enhance`: Disable post-swap face enhancement
- `--enhance-strength`: Post-swap enhancement strength
- `--crf`: Output video quality for x264; lower means higher quality, now defaulting to a higher-quality setting
- `--preset`: x264 speed/compression preset, now defaulting to `slower`
- `--audio-bitrate`: Output audio bitrate
- `--final-sharpen`: Mild whole-frame sharpening strength during final encoding; set `0` to disable
- `--swapper`: Optional explicit path to `inswapper_128.onnx`

## Notes

- The script clears `frames_in` and `frames_out` at the start of each run
- Processing is frame-by-frame, so runtime depends mainly on the total frame count
- The script prints input size, duration, estimated frame count, and extracted frame count to help diagnose unexpectedly slow runs
- A small MP4 file does not imply fast processing, because decoded frames are much larger than the compressed source video
- Even when CUDA is enabled, CPU usage can still be high because FFmpeg, image I/O, and frame orchestration still run mostly on CPU
- Final output now uses higher-quality `libx264` settings with configurable `crf`, `preset`, and mild whole-frame sharpening

## Common issues

### `ffmpeg is not available in PATH`

Install FFmpeg and make sure this command works:

```powershell
ffmpeg -version
```

### `model_file inswapper_128.onnx should exist`

The swapper model file is missing. Download `inswapper_128.onnx`, place it in one of the expected model directories, or pass it explicitly with `--swapper`.

### `CUDAExecutionProvider is unavailable`

The current Python environment did not load the GPU execution backend correctly. Check:

- whether `onnxruntime-gpu` is installed in the active environment
- whether `nvidia-smi` works normally
- whether the GPU driver and CUDA runtime are correctly installed

### Output looks soft or has ghosting

Try these first:

- use a clearer source face image
- increase `--det-size`
- lower `--crf`
- keep the built-in face enhancement enabled
- if you need another quality jump, add a restoration stage such as `CodeFormer` or `GFPGAN`
