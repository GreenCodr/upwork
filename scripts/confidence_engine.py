# scripts/confidence_engine.py

from typing import Optional, Union


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(x, hi))


def _safe_float(x: Union[float, int, str, None], default: float = 0.0) -> float:
    """
    Convert to float safely.
    If x is None or cannot be converted, return default.
    """
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def compute_confidence(
    duration_s: float,
    snr_db: Optional[float],
    speaker_similarity: Optional[float],
    device_match: Optional[float],
    history_count: int,
) -> float:
    """
    Production-grade confidence score for real human speech.
    Output range: [0.0 – 1.0]
    """

    # ---------------- Duration (20%) ----------------
    # 10s = minimum, 30s+ ideal
    duration_score = clamp((duration_s - 8.0) / 20.0)

    # ---------------- SNR (SOFT, 15%) ----------------
    # Speech SNR is usually 0–10 dB (do NOT punish)
    if snr_db is None:
        snr_score = 0.4
    elif snr_db <= 0:
        snr_score = 0.3
    elif snr_db < 10:
        snr_score = 0.3 + (snr_db / 10.0) * 0.4
    else:
        snr_score = 0.7

    # ---------------- Speaker similarity (30%) ----------------
    # Fix: speaker_similarity can be None if verification couldn't compute a score.
    # Keeping logic same: still clamps to [0, 1]. If None -> 0.0 (lowest confidence).
    speaker_score = clamp(_safe_float(speaker_similarity, default=0.0))

    # ---------------- Device consistency (15%) ----------------
    # Fix: device_match can also be None sometimes.
    device_score = clamp(_safe_float(device_match, default=0.0))

    # ---------------- History consistency (20%) ----------------
    if history_count >= 3:
        history_score = 1.0
    elif history_count == 2:
        history_score = 0.7
    elif history_count == 1:
        history_score = 0.4
    else:
        history_score = 0.2

    # ---------------- Final weighted confidence ----------------
    confidence = (
        0.30 * speaker_score +
        0.20 * duration_score +
        0.15 * snr_score +
        0.15 * device_score +
        0.20 * history_score
    )

    return round(clamp(confidence), 3)