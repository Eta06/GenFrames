"""
Video reading utilities using OpenCV and FFmpeg
"""

from typing import Iterator, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path


class VideoReader:
    """Video reader supporting multiple backends"""

    def __init__(
        self,
        video_path: str,
        backend: str = "opencv",
    ):
        """
        Initialize video reader

        Args:
            video_path: Path to input video
            backend: Backend to use ('opencv' or 'ffmpeg')
        """
        self.video_path = Path(video_path)
        self.backend = backend

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if backend == "opencv":
            self._init_opencv()
        elif backend == "ffmpeg":
            self._init_ffmpeg()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_opencv(self):
        """Initialize OpenCV video capture"""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))

    def _init_ffmpeg(self):
        """Initialize FFmpeg reader"""
        import ffmpeg

        # Probe video
        probe = ffmpeg.probe(str(self.video_path))
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

        self.width = int(video_info["width"])
        self.height = int(video_info["height"])

        # Parse fps
        fps_parts = video_info["r_frame_rate"].split("/")
        self.fps = float(fps_parts[0]) / float(fps_parts[1])

        self.frame_count = int(video_info.get("nb_frames", 0))

        # Create FFmpeg process
        self.process = (
            ffmpeg.input(str(self.video_path))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames"""
        if self.backend == "opencv":
            return self._iter_opencv()
        else:
            return self._iter_ffmpeg()

    def _iter_opencv(self) -> Iterator[np.ndarray]:
        """Iterate using OpenCV"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    def _iter_ffmpeg(self) -> Iterator[np.ndarray]:
        """Iterate using FFmpeg"""
        frame_size = self.width * self.height * 3

        while True:
            in_bytes = self.process.stdout.read(frame_size)
            if not in_bytes:
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            yield frame

    def read_frame(self) -> Optional[Tuple[bool, np.ndarray]]:
        """
        Read a single frame

        Returns:
            Tuple of (success, frame)
        """
        if self.backend == "opencv":
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, frame
        else:
            frame_size = self.width * self.height * 3
            in_bytes = self.process.stdout.read(frame_size)
            if not in_bytes:
                return False, None
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            return True, frame

    def get_fps(self) -> float:
        """Get video FPS"""
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution (width, height)"""
        return self.width, self.height

    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return self.frame_count

    def close(self):
        """Release resources"""
        if self.backend == "opencv":
            if hasattr(self, "cap"):
                self.cap.release()
        else:
            if hasattr(self, "process"):
                self.process.stdout.close()
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
