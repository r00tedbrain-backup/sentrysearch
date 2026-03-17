"""ffmpeg clip extraction."""

import subprocess


def trim_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
) -> None:
    """Extract a clip from a video file using ffmpeg.

    Uses input seeking (-ss before -i) for fast seeking.

    Args:
        video_path: Path to the source video.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Path for the output clip.
    """
    duration = end - start
    if duration <= 0:
        raise ValueError(f"End time ({end}) must be greater than start time ({start}).")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        check=True,
    )
