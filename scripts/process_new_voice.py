# scripts/process_new_voice.py

from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from scripts.audio_preprocess import normalize_audio   # 🔑 CRITICAL
from scripts.audio_quality import audio_quality_gate
from scripts.embed_ecapa import extract_embedding
from scripts.speaker_verification import speaker_verification_gate
from scripts.device_fingerprint import extract_device_fingerprint, device_match_score
from scripts.confidence_engine import compute_confidence
from scripts.version_decision import decide_voice_version
from scripts.user_registry import UserRegistry
from scripts.audio_utils import get_audio_duration


# ------------------ CONSTANTS ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_DURATION_SEC = 10.0

# ECAPA identity-grade threshold (REALISTIC)
STRICT_SPEAKER_THRESHOLD = 0.75


# ------------------ HELPERS ------------------

def _safe_round(x, ndigits: int):
    """Round safely: returns None if x is None or not a number."""
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return None


# ------------------ MAIN ENTRY ------------------

def process_new_voice(user_id: str, audio_path: str) -> dict:
    """
    Phase-2 backend: ECAPA-based identity verification
    Real-world safe (raw MP3/WAV supported)
    """

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"accepted": False, "reason": "Audio file not found"}

    user = UserRegistry(user_id)

    # ====================================================
    # 🔊 AUDIO NORMALIZATION (ABSOLUTELY REQUIRED)
    # ====================================================
    try:
        clean_audio = normalize_audio(audio_path)
    except Exception as e:
        return {
            "accepted": False,
            "reason": "Audio preprocessing failed",
            "error": str(e),
        }

    # ---------------- Duration ----------------
    duration = get_audio_duration(str(clean_audio))
    if duration < MIN_DURATION_SEC:
        return {
            "accepted": False,
            "reason": "Audio too short",
            "duration_sec": round(duration, 2),
        }

    # ---------------- Audio Quality (SOFT) ----------------
    quality = audio_quality_gate(str(clean_audio), dev_mode=True)
    soft_quality_fail = not quality["accepted"]

    # ---------------- ECAPA Embedding ----------------
    embedding = extract_embedding(clean_audio)
    embedding = embedding / np.linalg.norm(embedding)

    history_versions = user.get_versions()

    # ====================================================
    # 🧱 BASELINE BOOTSTRAP (FIRST VOICE ONLY)
    # ====================================================
    if not history_versions:
        version_id = str(int(datetime.now(timezone.utc).timestamp()))

        emb_dir = PROJECT_ROOT / "versions" / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        emb_path = emb_dir / f"{user_id}_{version_id}.npy"
        np.save(emb_path, embedding)

        user.add_voice_version(
            version_id=version_id,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path),   # 🔒 store ORIGINAL audio
            confidence=1.0,
            voice_type="RECORDED",
        )

        return {
            "accepted": True,
            "change_detected": False,
            "decision": {
                "action": "CREATE_BASELINE",
                "reason": "First voice stored",
            },
            "confidence": 1.0,
            "similarity": 1.0,
        }

    # ====================================================
    # 🔍 SPEAKER VERIFICATION (ECAPA)
    # ====================================================
    reference_embs = []
    for v in history_versions:
        p = v.get("embedding_path")
        if p:
            full = PROJECT_ROOT / p
            if full.exists():
                e = np.load(full).astype("float32")
                e = e / np.linalg.norm(e)
                reference_embs.append(e)

    speaker = speaker_verification_gate(
        new_emb=embedding,
        reference_embs=reference_embs,
        threshold=STRICT_SPEAKER_THRESHOLD,
    )

    # Pull similarity once and keep it consistent
    speaker_similarity = speaker.get("best_similarity", None)

    if not speaker["accepted"]:
        return {
            "accepted": False,
            "reason": "Different speaker detected",
            # Fix: best_similarity might be None, so round safely
            "similarity": _safe_round(speaker_similarity, 4),
        }

    # Fix: even if accepted, similarity could still be None in edge cases
    # (don’t crash; downstream logic already has None-guards)
    # ---------------- Device Fingerprint (SOFT) ----------------
    device_score = 1.0
    try:
        latest = user.get_latest_version()
        if latest and latest.get("audio_path"):
            fp_ref = extract_device_fingerprint(latest["audio_path"])
            fp_new = extract_device_fingerprint(str(audio_path))
            device_score = device_match_score(fp_new, fp_ref)
    except Exception:
        pass

    # ---------------- Confidence (ADVISORY ONLY) ----------------
    confidence = compute_confidence(
        duration_s=quality.get("duration", duration),
        # Pass through None if missing (your confidence_engine already handles None)
        snr_db=quality.get("snr_db", None),
        speaker_similarity=speaker_similarity,
        device_match=device_score,
        history_count=len(reference_embs),
    )

    if soft_quality_fail:
        confidence *= 0.6

    # ---------------- Decision ----------------
    decision = decide_voice_version(
        similarity=speaker_similarity,
        confidence=confidence,
        speaker_ok=True,
        device_match=device_score,
        embedding_path="N/A",
        audio_path=str(audio_path),
        user_dob=user.data.get("date_of_birth"),
    )

    # ---------------- Persist ----------------
    if decision.get("action") == "CREATE_VERSION":
        version_id = str(int(datetime.now(timezone.utc).timestamp()))

        emb_dir = PROJECT_ROOT / "versions" / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        emb_path = emb_dir / f"{user_id}_{version_id}.npy"
        np.save(emb_path, embedding)

        user.add_voice_version(
            version_id=version_id,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path),   # 🔒 ORIGINAL audio
            confidence=confidence,
            voice_type="RECORDED",
        )

    return {
        "accepted": True,
        "change_detected": False,
        "decision": decision,
        "confidence": round(confidence, 3),
        # Fix: speaker_similarity can be None → safe rounding
        "similarity": _safe_round(speaker_similarity, 4),
        "audio_quality_soft_fail": soft_quality_fail,
    }