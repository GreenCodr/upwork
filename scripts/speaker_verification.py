# scripts/speaker_verification.py

from __future__ import annotations

from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np


# ------------------ CORE MATH ------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Safe against zero norms / NaNs.
    """
    if a is None or b is None:
        return 0.0

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0

    sim = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(sim):
        return 0.0
    return sim


# ------------------ LOW-LEVEL GATE ------------------

def speaker_verification_gate(
    new_emb: np.ndarray,
    reference_embs: List[np.ndarray],
    threshold: float = 0.80
) -> Dict[str, Optional[float]]:
    """
    Low-level speaker verification gate.

    Returns:
      {
        "accepted": bool,
        "best_similarity": float,   # ALWAYS float (never None)
        "reason": str
      }
    """

    # If no references exist, we can't verify identity reliably.
    # Your pipeline creates a baseline earlier, so this should be rare.
    if not reference_embs:
        return {
            "accepted": False,
            "best_similarity": 0.0,
            "reason": "no_reference_embeddings",
        }

    best_sim = -1.0
    for ref in reference_embs:
        sim = cosine_similarity(new_emb, ref)
        if sim > best_sim:
            best_sim = sim

    # Ensure numeric
    if best_sim is None or not np.isfinite(best_sim):
        best_sim = 0.0
        return {
            "accepted": False,
            "best_similarity": 0.0,
            "reason": "invalid_similarity",
        }

    if best_sim < float(threshold):
        return {
            "accepted": False,
            "best_similarity": float(best_sim),
            "reason": "speaker_mismatch_detected",
        }

    return {
        "accepted": True,
        "best_similarity": float(best_sim),
        "reason": "accepted",
    }


# ------------------ OPTIONAL WRAPPER ------------------
# (Not used by your process_new_voice.py, but kept as a utility)

def verify_speaker(
    user_id: str,
    audio_or_embedding: Union[str, Path, np.ndarray],
    threshold: float = 0.80
) -> bool:
    """
    High-level speaker verification utility.

    - If audio_or_embedding is a numpy array -> treated as embedding
    - If it's a path/str -> embedding is extracted via scripts.embed_single_audio.extract_embedding

    NOTE: This wrapper loads embeddings from user's stored voice_versions.
    """

    # Lazy imports to avoid circular imports
    from scripts.user_registry import load_user

    user = load_user(user_id)

    reference_embs: List[np.ndarray] = []
    for v in user.get("voice_versions", []):
        p = v.get("embedding_path")
        if not p:
            continue
        try:
            reference_embs.append(np.load(p).astype("float32"))
        except Exception:
            continue

    # Get new embedding
    if isinstance(audio_or_embedding, np.ndarray):
        new_emb = audio_or_embedding.astype("float32")
    else:
        from scripts.embed_single_audio import extract_embedding
        new_emb = extract_embedding(str(audio_or_embedding))

    result = speaker_verification_gate(
        new_emb=new_emb,
        reference_embs=reference_embs,
        threshold=threshold,
    )

    return bool(result["accepted"])