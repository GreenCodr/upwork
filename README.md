---
title: upwork
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---
# 🎙️ Voice Evolution System

Automatic Voice Change Detection, Age-Based Playback & Future Voice Prediction

> This repository implements a real-world AI system that continuously tracks how a person’s voice evolves over time, detects meaningful changes automatically, and enables realistic playback of a voice at any age (past or predicted).

---

## Quick visual distinction
- All explanatory text (descriptions, features, architecture, notes) appears as normal Markdown content (headings, lists, paragraphs).
- All commands you should run in your terminal are shown in fenced code blocks labeled with the shell or language (e.g., bash, sh, python). This makes commands easy to spot and copy.

---

## Problem this project solves (text)
Human voices change naturally due to:
- Age
- Health
- Emotion
- Environment
- Recording devices

Today, there is no standard system that preserves a user's voice evolution intelligently. This project addresses that by:
- Automatically detecting significant voice changes
- Creating voice versions over time
- Allowing age-based playback like:
  - "Play my voice at age 8"
  - "How will my voice sound at 60?"
  - "Play my voice from 2015"

---

## Core Features (text)

### ✅ Phase 1 — Automatic Voice Change Detection
- Audio quality gating (duration, SNR)
- Speaker verification (ECAPA / wav2vec embeddings)
- Device fingerprint matching
- Confidence scoring
- FAISS similarity search
- Automatic version creation
- Version history storage

### ✅ Phase 2 — Age-Specific Voice Playback
- Voice timeline per user
- Age mapping using Date of Birth
- Closest-age voice selection
- SLERP interpolation between versions
- Past & future extrapolation
- Clear labeling:
  - RECORDED
  - INTERPOLATED
  - PREDICTED
- XTTS-based voice synthesis
- Rate limiting & audio caching
- Metadata tagging

### ✅ Phase 3 — Lightweight Learning (Optional)
- Builds an age-embedding dataset
- Tries learning age-to-voice deltas with small auxiliary models
- No heavy GPU training required
- Graceful fallback to rule-based logic when training data is insufficient

> Note: The system is intentionally robust even without training data.

---

## System Architecture (High level) (text)
Audio Input  
↓  
Quality Gate → Speaker Verification → Device Check  
↓  
Confidence Engine  
↓  
FAISS Similarity Search  
↓  
Version Decision Engine  
↓  
User Voice Timeline  
↓  
Playback Engine (Recorded / Interpolated / Predicted)

Frontend: Streamlit app for visualization and playback control.

---

## Streamlit Frontend (text)
The included Streamlit UI allows:
- User selection
- Voice timeline visualization
- Age-based voice playback
- Clear explanations of playback decisions
- Real-time synthesis output

To run the frontend, use the command below.

Commands
```bash
# Run the Streamlit frontend
streamlit run frontend/app.py
```

---

## Project Structure (text)
A simplified overview of the repo layout.

```text
voice-evolution/
├── frontend/          # Streamlit UI
├── scripts/           # Core system logic
├── config/            # Central config & thresholds
├── users/             # User profiles (runtime)
├── versions/          # Voice versions (runtime)
├── learning/          # Optional lightweight learning
├── src/               # API / core modules
├── README.md
└── .gitignore
```

---

## Use Cases (text)
- Personal voice archiving
- Voice aging research
- Speech therapy tracking
- Digital legacy preservation
- Forensic & historical voice analysis
- AI assistants with temporal voice memory

---

## How to Run the Voice Evolution System Locally

### 1️⃣ Prerequisites (text)
Ensure the following are installed on your system:
- Git
- Anaconda / Miniconda
- Python 3.9 or 3.10 (Conda recommended)
- FFmpeg (required for audio processing)

Commands for FFmpeg installation
```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg -y
```

---

### 2️⃣ Clone the repository (command)
```bash
git clone https://github.com/GreenCodr/voice-evolution-system.git
cd voice-evolution-system
```

---

### 3️⃣ Create & Activate Conda Environment (commands)
```bash
# Create the conda environment (example)
conda create -n voice-evo python=3.10 -y

# Activate the environment
conda activate voice-evo
```

---

### 4️⃣ Install Python dependencies (command)
```bash
pip install -r requirements.txt
```

---

### 5️⃣ Run the Frontend (recommended) (command)
```bash
streamlit run frontend/app.py
```

---

### 6️⃣ Run a backend test: age-based playback (command + example)
This short Python snippet demonstrates calling the playback service directly (backend-only):

```python
# Example: Run from the terminal with `python test_playback.py` or paste into an interpreter
from scripts.playback_service import play_voice

result = play_voice(
    user_id="user_002",
    target_age=60,
    text="Hello, this is how my voice may sound in the future."
)

print(result)
```

(Or run as a one-off here-document in bash:)

```bash
python - << 'PY'
from scripts.playback_service import play_voice

result = play_voice(
    user_id="user_002",
    target_age=60,
    text="Hello, this is how my voice may sound in the future."
)

print(result)
PY
```

Generated audio will be written to:
```text
outputs/
```

---

### 7️⃣ Creating a new user (text + example file path)
User profiles are stored under the users/ directory as JSON files:
- users/user_001.json
- users/user_002.json

Example: create a new user JSON file at users/user_003.json with the required fields (DOB, user_id, metadata).

---

## Important Notes (text)
- The system uses confidence-aware outputs and safe fallbacks to avoid hallucinated audio.
- Outputs, cache, and models are gitignored; check .gitignore for those paths.

Commands
```bash
# Example: reactivate environment reminder
conda activate voice-evo
```

---

Thank you for using the Voice Evolution System — commands are shown in fenced code blocks for easy copying, and all other information is presented as readable explanatory text.
