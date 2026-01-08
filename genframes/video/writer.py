"""
Video writing utilities using OpenCV and FFmpeg
"""

from typing import Optional
import cv2
import numpy as np
from pathlib import Path


class VideoWriter:
    """Video writer supporting multiple backends"""

    def __init__(
        self,
        output_path: str,
        fps: float,
        resolution: tuple[int, int],
        backend: str = "opencv",
        codec: str = "mp4v",
        crf: int = 18,
        preset: str = "medium",
    ):
        """
        Initialize video writer

        Args:
            output_path: Path to output video
            fps: Output video FPS
            resolution: Output resolution (width, height)
            backend: Backend to use ('opencv' or 'ffmpeg')
            codec: Video codec (for OpenCV: 'mp4v', 'h264', etc.)
            crf: Constant Rate Factor for quality (lower = better quality)
            preset: Encoding preset ('ultrafast', 'fast', 'medium', 'slow')
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width, self.height = resolution
        self.backend = backend
        self.codec = codec
        self.crf = crf
        self.preset = preset

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if backend == "opencv":
            self._init_opencv()
        elif backend == "ffmpeg":
            self._init_ffmpeg()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_opencv(self):
        """Initialize OpenCV video writer"""
        # Get fourcc code
        if self.codec == "mp4v":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif self.codec == "h264":
            fourcc = cv2.VideoWriter_fourcc(*"h264")
        elif self.codec == "x264":
            fourcc = cv2.VideoWriter_fourcc(*"x264")
        elif self.codec == "avc1":
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        else:
            # Try to use the codec string directly
            fourcc = cv2.VideoWriter_fourcc(*self.codec)

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.output_path}")

    def _init_ffmpeg(self):
        """Initialize FFmpeg writer"""
        import ffmpeg

        # Create FFmpeg process
        self.process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self.width}x{self.height}",
                r=self.fps,
            )
            .output(
                str(self.output_path),
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=self.crf,
                preset=self.preset,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame

        Args:
            frame: Frame to write (H, W, 3) in RGB format
        """
        if self.backend == "opencv":
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)
        else:
            # Write raw RGB data to FFmpeg
            self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        """Release resources"""
        if self.backend == "opencv":
            if hasattr(self, "writer"):
                self.writer.release()
        else:
            if hasattr(self, "process"):
                self.process.stdin.close()
                self.process.wait()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor"""
        self.close()
