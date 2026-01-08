"""
Basic usage example for GenFrames
"""

from genframes import FrameInterpolator

# Initialize interpolator (auto-selects best backend)
interpolator = FrameInterpolator(
    model="rife",
    model_version="rife-v4.6",
    backend="auto",  # Options: 'auto', 'mlx', 'mps', 'cuda', 'cpu'
)

# Process video with 2x interpolation (30fps → 60fps)
interpolator.process_video(
    input_path="input.mp4",
    output_path="output_2x.mp4",
    factor=2,
)

# Process video with 4x interpolation (30fps → 120fps)
interpolator.process_video(
    input_path="input.mp4",
    output_path="output_4x.mp4",
    factor=4,
)

# Get model information
info = interpolator.get_model_info()
print(f"Using: {info['backend']}")
