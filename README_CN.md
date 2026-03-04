# 视频换脸项目

这个目录现在是一个小型离线视频换脸项目，不再只是单文件示例。

## 文件说明

- `video_marge_demo.py`：项目入口脚本。
- `face_swap_project/cli.py`：命令行参数解析。
- `face_swap_project/pipeline.py`：主处理流程。
- `face_swap_project/insightface_runtime.py`：InsightFace 模型加载和运行时检查。
- `face_swap_project/enhancement.py`：换脸后的人脸增强逻辑。
- `face_swap_project/ffmpeg_tools.py`：拆帧、抽音频、合成视频。
- `face_swap_project/config.py`：配置对象定义。
- `requirements.txt`：这个 demo 需要的 Python 依赖。

## 环境要求

- Python 3.9
- Windows x64
- `ffmpeg` 已加入 `PATH`
- 有 NVIDIA 显卡时建议使用 `onnxruntime-gpu`，速度会更快

安装依赖：

```powershell
D:\python3\python.exe -m pip install -r .\requirements.txt
```

验证运行环境：

```powershell
D:\python3\python.exe -c "import insightface, onnxruntime as ort; print(insightface.__version__); print(ort.get_available_providers())"
ffmpeg -version
```

## 模型文件

当前项目需要以下模型：

- `buffalo_l`
- `inswapper_128.onnx`

说明：

- `buffalo_l` 一般会在首次运行时由 InsightFace 自动下载
- `inswapper_128.onnx` 需要你提前准备到本地

`inswapper_128.onnx` 可以放在下面任意一个位置：

- `.\inswapper_128.onnx`
- `C:\Users\HP\.insightface\models\inswapper_128.onnx`
- `C:\Users\HP\.insightface\models\inswapper_128\inswapper_128.onnx`

也可以在运行时通过 `--swapper` 显式传入。

## 当前增强方案

项目已经加入“换脸后做人脸增强”的步骤，处理顺序是：

1. 先执行 `inswapper_128.onnx` 换脸
2. 再对目标脸区域做局部对比度提升、双边滤波和锐化
3. 用软遮罩把增强结果融合回整帧

这套增强是轻量级方案，优点是部署简单，不额外依赖 PyTorch。它能改善糊脸和边缘发虚，但不等价于真正的人脸修复模型。

## 关于更强的模型

如果问题是“有没有比 `inswapper_128` 更强的效果方案”，结论要分开看：

- 在当前这个 InsightFace demo 路线里，`inswapper_128` 仍然是最常见、最容易落地的换脸模型。
- 想明显提升主观观感，通常更有效的做法不是单纯更换 swapper，而是在换脸后接人脸修复模型，例如 `CodeFormer` 或 `GFPGAN`。
- 这些方案属于额外的人脸增强/修复链路，不是当前项目默认直接集成的官方 InsightFace 替代 swapper。

也就是说，当前项目已经先实现了“轻量增强版本”；如果后面你要继续追求质量，可以再接入 `CodeFormer` 或 `GFPGAN` 作为第二阶段增强。

## CodeFormer 集成

项目现在已经支持把官方 `CodeFormer` 作为可选的第二阶段视频修复步骤使用。

你需要先在本地准备官方仓库：

```powershell
git clone https://github.com/sczhou/CodeFormer.git
```

然后按官方仓库要求安装它自己的依赖和权重。

运行时开启方式：

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\video_datas\minvideo1.mp4" --source ".\photo\OIP.webp" --out ".\video_result\result_codeformer.mp4" --device cuda --use-codeformer --codeformer-repo "D:\path\to\CodeFormer" --codeformer-python D:\python3\python.exe
```

相关参数：

- `--use-codeformer`：开启 CodeFormer 二阶段修复
- `--codeformer-repo`：本地官方 CodeFormer 仓库路径
- `--codeformer-python`：运行 CodeFormer 时使用的 Python
- `--codeformer-fidelity`：CodeFormer 的保真度参数，默认 `0.7`

## 运行方法

在当前目录下执行：

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result1.mp4" --device cuda
```

如果你想显式指定换脸模型路径：

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result1.mp4" --device cuda --swapper "C:\Users\HP\.insightface\models\inswapper_128\inswapper_128.onnx"
```

如果你想提高输出质量，可以这样运行：

```powershell
D:\python3\python.exe .\video_marge_demo.py --video ".\minvideo.mp4" --source ".\OIP.webp" --out ".\result_hq.mp4" --device cuda --det-size 1024 --crf 14 --preset slow
```

## 参数说明

- `--video`：输入视频路径
- `--source`：源人脸图片路径
- `--out`：输出视频路径
- `--workdir`：临时工作目录，默认是 `workdir_swap`
- `--fps`：可选，强制指定抽帧帧率；默认保持原视频帧率
- `--device`：`cpu` 或 `cuda`
- `--det-size`：检测分辨率，越大越稳，但越慢
- `--no-enhance`：关闭换脸后的人脸增强
- `--enhance-strength`：增强强度
- `--crf`：输出视频质量参数，值越小画质越高，文件也越大，当前默认更高质量
- `--preset`：x264 编码速度/压缩率预设，当前默认 `slower`
- `--audio-bitrate`：输出音频码率
- `--final-sharpen`：最终合成阶段的整帧轻度锐化强度，设为 `0` 可关闭
- `--swapper`：可选，手动指定 `inswapper_128.onnx` 路径

## 运行特点

- 每次运行开始时，脚本都会清空 `frames_in` 和 `frames_out`
- 处理方式是逐帧换脸，所以耗时主要取决于总帧数
- 脚本会打印输入视频大小、时长、预估帧数和实际抽帧数量，方便排查异常耗时
- MP4 文件体积小，不代表处理就快，因为拆成图片后数据量会明显变大
- 即使启用了 CUDA，CPU 占用依然可能很高，因为 `ffmpeg`、图片读写、逐帧调度这些步骤仍然主要在 CPU 上执行
- 最终输出现在使用更高质量的 `libx264` 编码参数，并加入 `crf`、`preset`、整帧轻锐化控制

## 常见问题

### `ffmpeg is not available in PATH`

说明系统找不到 `ffmpeg`。安装 FFmpeg 后，确认这条命令能正常输出版本：

```powershell
ffmpeg -version
```

### `model_file inswapper_128.onnx should exist`

说明换脸模型文件不存在。请下载 `inswapper_128.onnx`，放到约定目录，或者通过 `--swapper` 显式传入。

### `CUDAExecutionProvider is unavailable`

说明当前 Python 环境没有正确加载 GPU 推理后端。请检查：

- 当前环境里是否安装了 `onnxruntime-gpu`
- `nvidia-smi` 是否能正常工作
- 显卡驱动和 CUDA 运行环境是否正常

### 画面分辨率不高、重影或糊脸

可以优先尝试以下几种方式：

- 换更清晰的源脸图片
- 提高 `--det-size`
- 适当降低 `--crf`
- 保持换脸后的人脸增强开启
- 后续如需进一步提升，可再接入 `CodeFormer` 或 `GFPGAN`
