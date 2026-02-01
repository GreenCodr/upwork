# scripts/hybrid_playback_decider.py

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

# ------------------ PATH FIX ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------ IMPORTS ------------------
from scripts.user_registry import UserRegistry
from scripts.smart_version_selector import select_best_version
from scripts.age_selector import classify_age_relation

# ------------------ CONSTANTS ------------------
AGE_DELTAS_PATH = PROJECT_ROOT / "embeddings" / "age_deltas.npy"
EMB_DIR = PROJECT_ROOT / "versions" / "embeddings"


def _parse_recorded_date(recorded_utc: Optional[str]) -> Optional[datetime]:
    """
    Supports ISO format with 'Z' suffix.
    """
    if not recorded_utc:
        return None
    try:
        # examples: "2026-02-01T20:22:43.386675Z"
        s = recorded_utc.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _calculate_age(dob: Optional[str], recorded_dt: Optional[datetime]) -> Optional[int]:
    """
    dob format expected: YYYY-MM-DD
    """
    if not dob or not recorded_dt:
        return None
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d").date()
        d = recorded_dt.date()
        age = d.year - birth.year
        if (d.month, d.day) < (birth.month, birth.day):
            age -= 1
        # sanity clamp
        if age < 0 or age > 120:
            return None
        return age
    except Exception:
        return None


def _load_age_deltas() -> Optional[Dict[str, np.ndarray]]:
    """
    Loads embeddings/age_deltas.npy expecting a dict:
      {"children_to_adult": np.ndarray, "adult_to_children": np.ndarray}
    """
    if not AGE_DELTAS_PATH.exists():
        return None
    try:
        obj = np.load(AGE_DELTAS_PATH, allow_pickle=True).item()
        if not isinstance(obj, dict):
            return None
        # ensure arrays
        out: Dict[str, np.ndarray] = {}
        for k, v in obj.items():
            out[k] = np.asarray(v, dtype="float32")
        return out
    except Exception:
        return None


def decide_playback_mode(user_id: str, target_age: int) -> dict:
    """
    Phase-2 playback decision logic
    """
    user = UserRegistry(user_id)
    versions = user.get_versions()

    if not versions:
        return {"mode": "NONE", "reason": "no_voice_versions"}

    # 1) Select best recorded version
    selection = select_best_version(
        versions=versions,
        target_age=target_age
    )

    if selection.get("mode") == "RECORDED":
        return {
            "mode": "RECORDED",
            "version": selection["version"],
            "reason": "real_voice_close_to_target",
            "age_gap": selection.get("age_gap"),
        }

    # 2) Otherwise, attempt AGED playback from latest base version
    base_version = user.get_latest_version()
    if not base_version or not base_version.get("embedding_path"):
        return {"mode": "NONE", "reason": "no_embedding_available"}

    # --- Resolve base age robustly ---
    base_age = base_version.get("age_at_recording")

    if base_age is None:
        dob = None
        try:
            dob = user.data.get("date_of_birth")
        except Exception:
            dob = None

        recorded_dt = _parse_recorded_date(base_version.get("recorded_utc"))
        base_age = _calculate_age(dob, recorded_dt)

    # If still unknown, do a safe fallback (don’t produce UNKNOWN aging)
    if base_age is None:
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "missing_base_age_fallback_recorded",
        }

    relation = classify_age_relation(base_age, target_age)
    if relation == "same":
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "same_age_requested",
        }

    # Load base embedding
    try:
        base_emb = np.load(PROJECT_ROOT / base_version["embedding_path"]).astype("float32")
    except FileNotFoundError:
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "missing_base_embedding_fallback_recorded",
            "expected_path": str(PROJECT_ROOT / base_version["embedding_path"]),
        }

    base_emb = base_emb / (np.linalg.norm(base_emb) + 1e-12)

    # Load age deltas
    age_deltas = _load_age_deltas()
    if not age_deltas:
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "missing_age_deltas_fallback_recorded",
            "expected_path": str(AGE_DELTAS_PATH),
        }

    delta_key = "children_to_adult" if relation == "future" else "adult_to_children"
    if delta_key not in age_deltas:
        return {"mode": "RECORDED", "version": base_version, "reason": f"missing_delta:{delta_key}"}

    delta = age_deltas[delta_key]
    # Make sure dimensions match
    if delta.shape[0] != base_emb.shape[0]:
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "delta_dim_mismatch_fallback_recorded",
            "delta_dim": int(delta.shape[0]),
            "emb_dim": int(base_emb.shape[0]),
        }

    # Alpha based on age distance (cap at 40y)
    years = abs(int(base_age) - int(target_age))
    alpha = min(years / 40.0, 1.0)

    aged_emb = base_emb + alpha * delta
    aged_emb = aged_emb / (np.linalg.norm(aged_emb) + 1e-12)

    return {
        "mode": "AGED",
        "embedding": aged_emb,
        "base_version": base_version,  # REQUIRED by downstream
        "target_age": int(target_age),
        "alpha": round(float(alpha), 2),
        "relation": relation,
        "reason": "age_delta_applied",
    }