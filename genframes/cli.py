"""
Command-line interface for GenFrames
"""

import argparse
import sys
from pathlib import Path
from genframes import __version__
from genframes.interpolator import FrameInterpolator


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GenFrames - AI-powered video frame interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2x interpolation (30fps → 60fps)
  genframes input.mp4 output.mp4 --factor 2

  # 4x interpolation with MLX backend
  genframes input.mp4 output.mp4 --factor 4 --backend mlx

  # 8x interpolation with custom codec
  genframes input.mp4 output.mp4 --factor 8 --codec h264 --crf 20

  # Launch web UI
  genframes --ui
        """,
    )

    parser.add_argument("--version", action="version", version=f"GenFrames {__version__}")

    # UI mode
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch web UI instead of CLI mode",
    )

    # Video processing arguments
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Input video file",
    )

    parser.add_argument(
        "output",
        nargs="?",
        type=str,
        help="Output video file",
    )

    parser.add_argument(
        "-f",
        "--factor",
        type=int,
        default=2,
        choices=[2, 4, 8],
        help="Interpolation factor (default: 2)",
    )

    # Model options
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="rife",
        choices=["rife"],
        help="Model to use (default: rife)",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default="rife-v4.6",
        help="Model version (default: rife-v4.6)",
    )

    # Backend options
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "mlx", "mps", "cuda", "cpu"],
        help="Computation backend (default: auto)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific device to use",
    )

    # Video I/O options
    parser.add_argument(
        "--video-backend",
        type=str,
        default="opencv",
        choices=["opencv", "ffmpeg"],
        help="Video I/O backend (default: opencv)",
    )

    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Output video codec (default: mp4v)",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant Rate Factor for quality (lower = better, default: 18)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "slower"],
        help="Encoding preset (default: medium)",
    )

    args = parser.parse_args()

    # Launch UI mode
    if args.ui:
        launch_ui()
        return

    # Validate arguments for video mode
    if not args.input or not args.output:
        parser.error("Input and output video files are required (or use --ui for web interface)")

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize interpolator
        interpolator = FrameInterpolator(
            model=args.model,
            model_version=args.model_version,
            backend=args.backend,
            device=args.device,
        )

        # Process video
        interpolator.process_video(
            input_path=args.input,
            output_path=args.output,
            factor=args.factor,
            video_backend=args.video_backend,
            output_codec=args.codec,
            output_crf=args.crf,
            output_preset=args.preset,
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def launch_ui():
    """Launch Gradio web UI"""
    try:
        from genframes.ui.app import create_ui

        print("Launching GenFrames Web UI...")
        ui = create_ui()
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
        )
    except ImportError:
        print("Error: Gradio is required for web UI. Install with: pip install gradio", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error launching UI: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
