# frontend/app.py

import sys
import os
import json
import tempfile
from pathlib import Path

import streamlit as st

# ------------------ PATH FIX ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USERS_DIR = PROJECT_ROOT / "users"


def run_app():
    # ------------------ PAGE CONFIG ------------------
    st.set_page_config(
        page_title="Voice Evolution System",
        layout="centered"
    )

    # ==============================================================
    # HEADER
    # ==============================================================
    st.title("🎙️ Voice Evolution System")
    st.caption("Automatic voice change detection & age-based playback")
    st.divider()

    # ==============================================================
    # USER SELECTION
    # ==============================================================
    st.header("👤 User Dashboard")

    user_files = list(USERS_DIR.glob("*.json"))
    if not user_files:
        st.error("No users found. Please create a user first.")
        st.stop()

    user_ids = [f.stem for f in user_files]
    selected_user = st.selectbox("Select User", user_ids)

    user_path = USERS_DIR / f"{selected_user}.json"
    user = json.loads(user_path.read_text())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("User ID", user["user_id"])
        st.metric("Date of Birth", user.get("date_of_birth", "Unknown"))
    with col2:
        st.metric("Total Voice Versions", len(user.get("voice_versions", [])))
        st.metric("Account Created", user.get("created_utc", "")[:10])

    st.divider()

    # ==============================================================
    # PHASE 1 — VOICE INGESTION
    # ==============================================================
    st.header("🎙️ Upload Voice Sample")

    uploaded = st.file_uploader(
        "Upload voice sample (WAV / MP3, minimum 10 seconds)",
        type=["wav", "mp3"]
    )

    if uploaded:
        st.audio(uploaded)
        st.success("Voice file received ✔️")

        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing voice sample..."):
            from scripts.process_new_voice import process_new_voice
            result = process_new_voice(
                user_id=selected_user,
                audio_path=tmp_path
            )

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if not result.get("accepted", False):
            st.error(f"❌ {result.get('reason', 'Rejected')}")
        else:
            decision = result.get("decision", {})

            st.success("Voice analyzed successfully")
            st.write(f"• Change detected: `{result.get('change_detected')}`")
            st.write(f"• Decision: `{decision.get('action')}`")

            if "reason" in decision:
                st.write(f"• Reason: {decision['reason']}")

            st.metric("Confidence", result.get("confidence", 0.0))
            st.metric("Similarity", result.get("similarity", 0.0))

            if result.get("audio_quality_soft_fail"):
                st.warning("⚠️ Audio quality was suboptimal (soft penalty applied)")
    else:
        st.info("Waiting for voice input...")

    st.divider()

    # ==============================================================
    # PHASE 2 — AGE-BASED PLAYBACK
    # ==============================================================
    st.header("🎧 Age-Based Voice Playback")

    target_age = st.slider("Select target age", 5, 90, 60)

    text_to_speak = st.text_area(
        "Text to speak",
        value="Hello, this is how my voice may sound in the future.",
        height=180
    )

    if st.button("▶️ Play Voice"):
        with st.spinner("Preparing voice playback..."):
            from scripts.playback_service import play_voice
            result = play_voice(
                user_id=selected_user,
                target_age=target_age,
                text=text_to_speak
            )

        if result["mode"] == "ERROR":
            st.error(result["reason"])
        else:
            st.audio(result["audio_path"])

    st.divider()
    st.caption("Voice Evolution System — Phase 2 complete")