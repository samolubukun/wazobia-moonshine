---
title: WaZoBia Moonshine ASR Engine
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: WaZoBia Moonshine-Tiny Automatic Speech Recognition Studio
---

# 🎙️ WaZoBia Moonshine ASR Engine: Sovereign Nigerian Speech Recognition

A production Gradio speech-to-text studio serving all **6 fine-tuned Moonshine-Tiny ASR models** for Nigerian Sovereign Languages: **Hausa**, **Yorùbá**, and **Igbo**.

Designed to run efficiently on **Hugging Face ZeroGPU Spaces** and **local environments** (CPU or GPU).

---

## 🌟 The 6 Fine-Tuned Models (Stage 1 & Stage 2/3)

Moonshine-Tiny (~27M parameters) provides real-time, low-latency automatic speech transcription adapted across two distinct training milestones:

### 🟡 Stage 1: Foundational Acoustic Models
Trained on studio-quality scripture audio for acoustic stability, robust dictionary alignment, and baseline phonetic accuracy.
- **Hausa (Foundation)**: [`samuelolubukun/moonshine-tiny-hausa-openbible`](https://huggingface.co/samuelolubukun/moonshine-tiny-hausa-openbible)
- **Yorùbá (Foundation)**: [`samuelolubukun/moonshine-tiny-yoruba-openbible`](https://huggingface.co/samuelolubukun/moonshine-tiny-yoruba-openbible)
- **Igbo (Foundation)**: [`samuelolubukun/moonshine-tiny-igbo-openbible`](https://huggingface.co/samuelolubukun/moonshine-tiny-igbo-openbible)

### 🟢 Stage 2 & 3: Conversational Fine-Tuned Models (Recommended)
Further adapted on spontaneous Nigerian voices (Google FLEURS + NaijaVoices) to handle background acoustic variance, colloquial accents, conversational cadence, and informal phrasing.
- **Hausa (v2 Conversational)**: [`samuelolubukun/moonshine-tiny-hausa-v2`](https://huggingface.co/samuelolubukun/moonshine-tiny-hausa-v2)
- **Yorùbá (v2 Conversational)**: [`samuelolubukun/moonshine-tiny-yoruba-v2`](https://huggingface.co/samuelolubukun/moonshine-tiny-yoruba-v2)
- **Igbo (v2 Conversational)**: [`samuelolubukun/moonshine-tiny-igbo-v2`](https://huggingface.co/samuelolubukun/moonshine-tiny-igbo-v2)

---

## 🎧 Recognition Features

- **Microphone & File Upload**: Real-time transcription from microphone or existing `.wav`/`.mp3` audio.
- **Configurable Beam Search Width**: Adjust beam exploration (1–10) to balance inference speed and decoding accuracy.
- **Repetition Penalty**: Controls n-gram repeat suppression for crisp, concise transcripts.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/samolubukun/wazobia-moonshine.git
cd wazobia-moonshine

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio application
python app.py
```

The web application will open at `http://localhost:7860`.

---

## ⚖️ Datasets, Attributions & Citations

1. **Base Architecture**: Useful Sensors Moonshine ASR ([Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)).
2. **Datasets**:
   - **OpenBible Project**: Scripture narration corpus for Stage 1 acoustic foundation.
   - **NaijaVoices Community** (*Emezue et al., 2025*): Natural conversational African speech corpus ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
   - **Google FLEURS**: Nigerian language speech benchmarks ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
3. **Fine-Tuning & Engineering**: [Samuel Olubukun](https://github.com/samolubukun).
