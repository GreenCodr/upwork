import subprocess
from pathlib import Path
import uuid
import os


def normalize_audio(input_path: str) -> str:
    """
    Converts any audio to clean WAV (16kHz, mono, PCM).
    Returns path to cleaned wav.
    """

    input_path = Path(input_path)
    out_path = input_path.parent / f"clean_{uuid.uuid4().hex}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    # ✅ NEW: verify FFmpeg success
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    # ✅ NEW: verify output exists
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced empty or missing output file")

    return str(out_path)