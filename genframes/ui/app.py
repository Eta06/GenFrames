"""
Gradio web UI for GenFrames
"""

import gradio as gr
import tempfile
from pathlib import Path
from genframes import __version__
from genframes.interpolator import FrameInterpolator
from genframes.backends.base import detect_best_backend


def process_video_ui(
    input_video,
    factor,
    model_version,
    backend,
    video_backend,
    codec,
    crf,
    preset,
    progress=gr.Progress(),
):
    """
    Process video through UI

    Args:
        input_video: Input video file
        factor: Interpolation factor
        model_version: Model version to use
        backend: Computation backend
        video_backend: Video I/O backend
        codec: Output codec
        crf: Output quality
        preset: Encoding preset
        progress: Gradio progress tracker

    Returns:
        Output video path and status message
    """
    if input_video is None:
        return None, "❌ Please upload a video file"

    try:
        progress(0, desc="Initializing...")

        # Create output file
        output_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4", prefix="genframes_"
        )
        output_path = output_file.name
        output_file.close()

        # Initialize interpolator
        interpolator = FrameInterpolator(
            model="rife",
            model_version=model_version,
            backend=backend,
        )

        # Progress callback
        def update_progress(current, total):
            progress(current / total, desc=f"Interpolating frames: {current}/{total}")

        # Process video
        interpolator.process_video(
            input_path=input_video,
            output_path=output_path,
            factor=factor,
            video_backend=video_backend,
            output_codec=codec,
            output_crf=crf,
            output_preset=preset,
            progress_callback=update_progress,
        )

        progress(1.0, desc="Complete!")

        status = f"✓ Successfully interpolated video with {factor}x factor"
        return output_path, status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_system_info():
    """Get system information"""
    backend = detect_best_backend()
    from genframes.backends.base import get_backend

    backend_obj = get_backend(backend)
    device_name = backend_obj.get_device_name()

    return f"""
### System Information
- **Best Backend**: {backend}
- **Device**: {device_name}
- **Version**: GenFrames {__version__}
    """


def create_ui():
    """Create Gradio UI"""

    with gr.Blocks(
        title="GenFrames - Video Frame Interpolation",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🎬 GenFrames - AI Video Frame Interpolation

            Make your videos buttery smooth with AI-powered frame interpolation!
            Upload a video and increase its frame rate by 2x, 4x, or 8x.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Input")

                input_video = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                )

                factor = gr.Radio(
                    choices=[2, 4, 8],
                    value=2,
                    label="Interpolation Factor",
                    info="2x: 30fps→60fps | 4x: 30fps→120fps | 8x: 30fps→240fps",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    model_version = gr.Dropdown(
                        choices=["rife-v4.6", "rife-v4.15-lite"],
                        value="rife-v4.6",
                        label="Model Version",
                        info="v4.6 = better quality, lite = faster",
                    )

                    backend = gr.Dropdown(
                        choices=["auto", "mlx", "mps", "cuda", "cpu"],
                        value="auto",
                        label="Backend",
                        info="Auto = automatically select best available",
                    )

                    video_backend = gr.Dropdown(
                        choices=["opencv", "ffmpeg"],
                        value="opencv",
                        label="Video Backend",
                    )

                    codec = gr.Dropdown(
                        choices=["mp4v", "h264", "x264", "avc1"],
                        value="mp4v",
                        label="Output Codec",
                    )

                    crf = gr.Slider(
                        minimum=0,
                        maximum=51,
                        value=18,
                        step=1,
                        label="Quality (CRF)",
                        info="Lower = better quality, larger file",
                    )

                    preset = gr.Dropdown(
                        choices=["ultrafast", "fast", "medium", "slow", "slower"],
                        value="medium",
                        label="Encoding Preset",
                        info="Slower = better compression",
                    )

                process_btn = gr.Button(
                    "🚀 Interpolate Video",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("## Output")

                output_video = gr.Video(
                    label="Interpolated Video",
                    interactive=False,
                )

                status_text = gr.Markdown("Ready to process")

                gr.Markdown(
                    """
                    ### How it works

                    1. **Upload** your video
                    2. **Choose** interpolation factor (2x, 4x, or 8x)
                    3. **Click** "Interpolate Video"
                    4. **Download** your smooth video!

                    GenFrames uses the RIFE model to generate intermediate frames,
                    creating smooth motion between original frames.
                    """
                )

        # System info
        gr.Markdown(get_system_info())

        # Examples
        gr.Markdown("### Examples")
        gr.Markdown(
            """
            - **2x (30fps → 60fps)**: Standard smooth motion, best for most videos
            - **4x (30fps → 120fps)**: Very smooth, great for slow motion effects
            - **8x (30fps → 240fps)**: Ultra smooth, perfect for detailed analysis
            """
        )

        # Process button click
        process_btn.click(
            fn=process_video_ui,
            inputs=[
                input_video,
                factor,
                model_version,
                backend,
                video_backend,
                codec,
                crf,
                preset,
            ],
            outputs=[output_video, status_text],
        )

        gr.Markdown(
            """
            ---

            Made with ❤️ using [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
            and optimized for Apple Silicon with MLX
            """
        )

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch()
