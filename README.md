# GenFrames 🎬

AI-powered video frame interpolation tool that makes your videos buttery smooth! Generate intermediate frames to increase frame rates using state-of-the-art RIFE model.

## Features ✨

- **Multiple Interpolation Modes**: 2x, 4x, 8x frame rate increase
  - 30fps → 60fps (2x)
  - 30fps → 120fps (4x)
  - 30fps → 240fps (8x)
- **Optimized for Apple Silicon**: Native MLX and PyTorch MPS support
- **Multi-Backend**: Automatic detection and selection of best available backend
  - MLX (Apple Silicon - fastest on M1/M2/M3/M4/M5)
  - PyTorch MPS (Apple Silicon)
  - PyTorch CUDA (NVIDIA GPUs)
  - PyTorch CPU (fallback)
- **RIFE Model**: State-of-the-art Real-Time Intermediate Flow Estimation
- **Easy to Use**: CLI tool + Web UI
- **Automatic Model Download**: Models downloaded on first use

## Installation

### Basic Installation

```bash
pip install -e .
```

### With All Features

```bash
pip install -e ".[all]"
```

### Platform-Specific

**Apple Silicon (MLX + PyTorch MPS)**:
```bash
pip install -e ".[mlx,torch,ui,ffmpeg]"
```

**NVIDIA GPU (CUDA)**:
```bash
pip install -e ".[torch,ui,ffmpeg]"
```

## Quick Start

### CLI Usage

```bash
# 2x interpolation (30fps → 60fps)
genframes input.mp4 output.mp4 --factor 2

# 4x interpolation (30fps → 120fps)
genframes input.mp4 output.mp4 --factor 4

# 8x interpolation with specific backend
genframes input.mp4 output.mp4 --factor 8 --backend mlx
```

### Web UI

```bash
genframes --ui
```

Then open your browser to `http://localhost:7860`

### Python API

```python
from genframes import FrameInterpolator

# Initialize interpolator (auto-selects best backend)
interpolator = FrameInterpolator(model="rife", backend="auto")

# Interpolate video
interpolator.process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    factor=2  # 2x frame rate
)
```

## How It Works

GenFrames uses the RIFE (Real-Time Intermediate Flow Estimation) model to generate intermediate frames:

```
Original: Frame 1 -----------> Frame 2
                    ↓
With 2x:  Frame 1 → Gen 1.5 → Frame 2
                    ↓
With 4x:  Frame 1 → Gen 1.25 → Gen 1.5 → Gen 1.75 → Frame 2
```

## Supported Models

- **RIFE 4.6** (default) - Fast and high quality
- More models coming soon (FILM, etc.)

## System Requirements

- Python 3.9+
- 8GB+ RAM recommended
- For best performance:
  - Apple Silicon: M1 or newer
  - NVIDIA: GPU with 4GB+ VRAM
  - AMD: GPU with 4GB+ VRAM

## License

MIT License

## Acknowledgments

- [RIFE](https://github.com/hzwer/ECCV2022-RIFE) - Real-Time Intermediate Flow Estimation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
