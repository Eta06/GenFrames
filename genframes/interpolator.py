"""
Main frame interpolator orchestrator
"""

from typing import Optional, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm

from genframes.backends.base import get_backend
from genframes.models.rife import RIFEModel
from genframes.video.reader import VideoReader
from genframes.video.writer import VideoWriter


class FrameInterpolator:
    """Main class for video frame interpolation"""

    def __init__(
        self,
        model: str = "rife",
        model_version: str = "rife-v4.6",
        backend: str = "auto",
        device: Optional[str] = None,
    ):
        """
        Initialize frame interpolator

        Args:
            model: Model to use ('rife' currently supported)
            model_version: Specific model version
            backend: Backend to use ('auto', 'mlx', 'mps', 'cuda', 'cpu')
            device: Specific device to use
        """
        self.model_name = model
        self.model_version = model_version

        # Initialize backend
        print(f"Initializing backend: {backend}")
        self.backend = get_backend(backend, device)
        self.backend.initialize()
        print(f"Using: {self.backend.get_device_name()}")

        # Initialize model
        if model == "rife":
            self.model = RIFEModel(
                backend=self.backend,
                model_version=model_version,
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def process_video(
        self,
        input_path: str,
        output_path: str,
        factor: int = 2,
        video_backend: str = "opencv",
        output_codec: str = "mp4v",
        output_crf: int = 18,
        output_preset: str = "medium",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Process video and interpolate frames

        Args:
            input_path: Path to input video
            output_path: Path to output video
            factor: Interpolation factor (2, 4, or 8)
            video_backend: Video I/O backend ('opencv' or 'ffmpeg')
            output_codec: Output video codec
            output_crf: Output video quality (lower = better)
            output_preset: Encoding preset
            progress_callback: Optional callback for progress updates (current, total)
        """
        if factor not in [2, 4, 8]:
            raise ValueError(f"Invalid factor: {factor}. Must be 2, 4, or 8")

        print(f"\n{'='*60}")
        print(f"GenFrames - Video Frame Interpolation")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Factor: {factor}x")
        print(f"Model: {self.model_version}")
        print(f"Backend: {self.backend.get_device_name()}")
        print(f"{'='*60}\n")

        # Open input video
        reader = VideoReader(input_path, backend=video_backend)
        input_fps = reader.get_fps()
        output_fps = input_fps * factor
        width, height = reader.get_resolution()
        total_frames = reader.get_frame_count()

        print(f"Input: {width}x{height} @ {input_fps:.2f} fps ({total_frames} frames)")
        print(f"Output: {width}x{height} @ {output_fps:.2f} fps (~{total_frames * factor} frames)")
        print()

        # Open output video
        writer = VideoWriter(
            output_path,
            fps=output_fps,
            resolution=(width, height),
            backend=video_backend,
            codec=output_codec,
            crf=output_crf,
            preset=output_preset,
        )

        try:
            # Calculate number of intermediate frames per pair
            frames_per_pair = factor - 1

            # Read first frame
            ret, prev_frame = reader.read_frame()
            if not ret:
                raise RuntimeError("Failed to read first frame")

            # Write first frame
            writer.write_frame(prev_frame)

            # Process video
            frame_count = 1
            output_count = 1

            with tqdm(total=total_frames, desc="Interpolating", unit="frame") as pbar:
                while True:
                    # Read next frame
                    ret, curr_frame = reader.read_frame()
                    if not ret:
                        break

                    # Generate intermediate frames
                    intermediate_frames = self._generate_intermediate_frames(
                        prev_frame,
                        curr_frame,
                        frames_per_pair,
                    )

                    # Write intermediate frames
                    for inter_frame in intermediate_frames:
                        writer.write_frame(inter_frame)
                        output_count += 1

                    # Write current frame
                    writer.write_frame(curr_frame)
                    output_count += 1

                    # Update
                    prev_frame = curr_frame
                    frame_count += 1
                    pbar.update(1)

                    if progress_callback:
                        progress_callback(frame_count, total_frames)

        finally:
            reader.close()
            writer.close()

        print(f"\n{'='*60}")
        print(f"✓ Interpolation complete!")
        print(f"  Input frames: {frame_count}")
        print(f"  Output frames: {output_count}")
        print(f"  Saved to: {output_path}")
        print(f"{'='*60}\n")

    def _generate_intermediate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """
        Generate intermediate frames between two frames

        Args:
            frame1: First frame
            frame2: Second frame
            num_frames: Number of intermediate frames

        Returns:
            List of intermediate frames
        """
        if num_frames == 0:
            return []
        elif num_frames == 1:
            # Simple case: generate single middle frame
            return [self.model.interpolate(frame1, frame2, timestep=0.5)]
        else:
            # Recursive interpolation for 4x and 8x
            return self.model.interpolate_recursive(frame1, frame2, num_frames)

    def interpolate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediate: int = 1,
    ) -> list[np.ndarray]:
        """
        Interpolate frames between two input frames (API method)

        Args:
            frame1: First frame (H, W, 3) RGB [0, 255]
            frame2: Second frame (H, W, 3) RGB [0, 255]
            num_intermediate: Number of intermediate frames

        Returns:
            List of all frames including input frames
        """
        self.model.ensure_model_loaded()

        intermediate = self._generate_intermediate_frames(frame1, frame2, num_intermediate)

        # Return all frames in sequence
        return [frame1] + intermediate + [frame2]

    def get_model_info(self) -> dict:
        """
        Get information about loaded model

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model_name,
            "version": self.model_version,
            "backend": self.backend.get_device_name(),
            "device": str(self.backend.device),
        }
