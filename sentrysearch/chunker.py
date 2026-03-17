"""Video chunking logic."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def _get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def chunk_video(
    video_path: str,
    chunk_duration: int = 30,
    overlap: int = 5,
) -> list[dict]:
    """Split a video into overlapping chunks using ffmpeg.

    Args:
        video_path: Path to the input mp4 file.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of dicts with keys: chunk_path, source_file, start_time, end_time.
    """
    video_path = str(Path(video_path).resolve())
    duration = _get_video_duration(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_")
    step = chunk_duration - overlap
    chunks = []

    if duration <= chunk_duration:
        chunk_path = os.path.join(tmp_dir, "chunk_000.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", "0",
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )
        return [
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": 0.0,
                "end_time": duration,
            }
        ]

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        t = end - start
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.mp4")

        # Input seeking (-ss before -i) for fast seek
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(t),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )

        chunks.append(
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": start,
                "end_time": end,
            }
        )

        start += step
        idx += 1

        # Avoid a tiny trailing chunk that's entirely within the previous chunk
        if start + overlap >= duration:
            break

    return chunks


def scan_directory(directory_path: str) -> list[str]:
    """Recursively find all .mp4 files in a directory.

    Args:
        directory_path: Root directory to scan.

    Returns:
        Sorted list of absolute file paths.
    """
    mp4_files = []
    for root, _dirs, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, f))
    mp4_files.sort()
    return mp4_files
