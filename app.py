"""WaZoBia Moonshine-Tiny ASR Serving Engine for Hugging Face ZeroGPU Spaces.

Features all 6 Sovereign Speech Recognition Models:
  1. samuelolubukun/moonshine-tiny-hausa-v2 (Hausa v2 FLEURS + NaijaVoices)
  2. samuelolubukun/moonshine-tiny-hausa-openbible (Hausa Stage 1 Foundation)
  3. samuelolubukun/moonshine-tiny-igbo-v2 (Igbo v2 FLEURS + NaijaVoices)
  4. samuelolubukun/moonshine-tiny-igbo-openbible (Igbo Stage 1 Foundation)
  5. samuelolubukun/moonshine-tiny-yoruba-v2 (Yoruba v2 FLEURS + NaijaVoices)
  6. samuelolubukun/moonshine-tiny-yoruba-openbible (Yoruba Stage 1 Foundation)

Provides:
  - Interactive Gradio Web UI with audio recording, player, and parameter tuning.
  - OpenAI-compatible REST API (/v1/audio/transcriptions, /v1/models, /healthz).
  - Hugging Face ZeroGPU compatibility via @spaces.GPU.
"""

import hmac
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr
import gradio_client.utils
import spaces

# Fix Gradio schema introspection bug
_orig_get_type = gradio_client.utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _orig_get_type(schema)
gradio_client.utils.get_type = _patched_get_type

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from huggingface_hub import hf_hub_download
from starlette.routing import BaseRoute, Match
from transformers import (
    MoonshineForConditionalGeneration,
    PreTrainedTokenizerFast,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Moonshine-WaZoBia-ASR")

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODELS_CATALOG = {
    "hausa-v2": "samuelolubukun/moonshine-tiny-hausa-v2",
    "hausa-openbible": "samuelolubukun/moonshine-tiny-hausa-openbible",
    "yoruba-v2": "samuelolubukun/moonshine-tiny-yoruba-v2",
    "yoruba-openbible": "samuelolubukun/moonshine-tiny-yoruba-openbible",
    "igbo-v2": "samuelolubukun/moonshine-tiny-igbo-v2",
    "igbo-openbible": "samuelolubukun/moonshine-tiny-igbo-openbible",
}

MODEL_ID_MAP = {
    "samuelolubukun/moonshine-tiny-hausa-v2": "hausa-v2",
    "samuelolubukun/moonshine-tiny-hausa-openbible": "hausa-openbible",
    "samuelolubukun/moonshine-tiny-igbo-v2": "igbo-v2",
    "samuelolubukun/moonshine-tiny-igbo-openbible": "igbo-openbible",
    "samuelolubukun/moonshine-tiny-yoruba-v2": "yoruba-v2",
    "samuelolubukun/moonshine-tiny-yoruba-openbible": "yoruba-openbible",
    "hausa": "hausa-v2",
    "ha": "hausa-v2",
    "igbo": "igbo-v2",
    "ig": "igbo-v2",
    "yoruba": "yoruba-v2",
    "yo": "yoruba-v2",
}

DEFAULT_MODEL_KEY = "hausa-v2"

# In-memory storage for lazy loading
loaded_models: Dict[str, MoonshineForConditionalGeneration] = {}
loaded_processors: Dict[str, Wav2Vec2Processor] = {}

def get_model_and_processor(model_key: str):
    """Lazy-load Moonshine model and tokenizer on demand."""
    if model_key not in MODELS_CATALOG:
        model_key = DEFAULT_MODEL_KEY
    repo_id = MODELS_CATALOG[model_key]

    if model_key not in loaded_models:
        logger.info(f"Loading Moonshine model: {repo_id}")
        tok_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=False,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        model = MoonshineForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()

        loaded_processors[model_key] = processor
        loaded_models[model_key] = model
        logger.info(f"Successfully loaded {repo_id}")

    return loaded_models[model_key], loaded_processors[model_key], repo_id

# ---------------------------------------------------------------------------
# ZeroGPU Inference Function
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def gpu_transcribe(
    audio_wav: np.ndarray,
    model_key: str,
    num_beams: int = 5,
    repetition_penalty: float = 1.1,
) -> Dict[str, Any]:
    model, processor, repo_id = get_model_and_processor(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        model = model.to(dtype)

    inputs = processor(audio_wav, sampling_rate=16000, return_tensors="pt").to(device)
    token_limit = min(200, max(24, int((len(audio_wav) / 16000) * 7.5) + 6))

    with torch.inference_mode():
        gen_ids = model.generate(
            inputs["input_values"].to(dtype),
            attention_mask=inputs.get("attention_mask"),
            decoder_start_token_id=1,
            eos_token_id=2,
            max_new_tokens=token_limit,
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

    transcription = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
    duration = round(float(len(audio_wav)) / 16000.0, 2)

    return {
        "text": transcription,
        "duration": duration,
        "model": repo_id,
        "model_key": model_key,
    }

def decode_audio_bytes(data: bytes) -> np.ndarray:
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception:
        wav, sr = librosa.load(io.BytesIO(data), sr=16000)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)

# ---------------------------------------------------------------------------
# FastAPI Endpoints (OpenAI-compatible)
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(
    title="WaZoBia Moonshine ASR Serving API",
    description="OpenAI-compatible transcription endpoint hosting all 6 Moonshine ASR models for Hausa, Igbo, and Yoruba.",
    version="1.0.0",
)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_scheme = APIKeyHeader(name="Authorization", scheme_name="BearerAuth", auto_error=False)

def verify_api_key(auth_header: Optional[str] = Security(_api_key_scheme)):
    expected = os.environ.get("MOONSHINE_API_KEY")
    if not expected:
        return True  # Open access if secret not configured
    token = ""
    if auth_header:
        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else auth_header.strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return True

@fastapi_app.get("/healthz")
def health_check():
    return {
        "status": "healthy",
        "service": "wazobia-moonshine-asr",
        "supported_models": list(MODELS_CATALOG.values()),
        "time": datetime.utcnow().isoformat(),
    }

@fastapi_app.get("/v1/models")
def list_models(auth=Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": repo,
                "object": "model",
                "owned_by": "samuelolubukun",
                "type": "automatic-speech-recognition",
            }
            for repo in MODELS_CATALOG.values()
        ],
    }

@fastapi_app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    auth=Depends(verify_api_key),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file empty.")

    key = DEFAULT_MODEL_KEY
    if model and model.strip() in MODEL_ID_MAP:
        key = MODEL_ID_MAP[model.strip()]
    elif language and language.strip().lower() in MODEL_ID_MAP:
        key = MODEL_ID_MAP[language.strip().lower()]

    wav = decode_audio_bytes(content)
    try:
        result = await run_in_threadpool(gpu_transcribe, wav, key)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if response_format == "text":
        return PlainTextResponse(result["text"])
    return JSONResponse(result)

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
def gradio_transcribe(audio_path, model_choice, num_beams, repetition_penalty):
    if not audio_path:
        return "⚠️ Please upload or record an audio sample."
    wav, sr = librosa.load(audio_path, sr=16000)
    key = MODEL_ID_MAP.get(model_choice, DEFAULT_MODEL_KEY)
    res = gpu_transcribe(wav, key, num_beams=num_beams, repetition_penalty=repetition_penalty)
    return f"""### 📝 Transcription Result ({res['duration']}s)
> **{res['text']}**

---
- **Model Repo:** `{res['model']}`
- **Beam Width:** `{num_beams}` | **Repetition Penalty:** `{repetition_penalty}`
"""

with gr.Blocks(title="WaZoBia Moonshine-Tiny ASR Hub") as demo:
    gr.Markdown("""# 🎙️ WaZoBia Moonshine-Tiny ASR Hub
### High-Performance Sovereign Speech-to-Text for Hausa, Igbo, and Yoruba (All 6 Models)
Hosted on Hugging Face ZeroGPU with an **OpenAI-compatible REST API** (`/v1/audio/transcriptions`).
""")

    with gr.Row():
        with gr.Column(scale=3):
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Speech (Hausa, Igbo, or Yoruba)")
            transcribe_btn = gr.Button("⚡ Transcribe Audio", variant="primary")
            transcription_out = gr.Markdown("### Transcription will appear here...")

        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS_CATALOG.values()),
                value=MODELS_CATALOG["hausa-v2"],
                label="Select Moonshine Model",
            )
            beams_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Search Width")
            rep_slider = gr.Slider(minimum=1.0, maximum=1.5, value=1.1, step=0.05, label="Repetition Penalty")
            gr.Markdown("""
#### 💡 Model Breakdown:
- **`*-v2` (Stage 3)**: Optimized for conversational, spontaneous Nigerian speech (FLEURS + NaijaVoices).
- **`*-openbible` (Stage 1)**: Foundational acoustic model trained on studio scripture narration.
""")

    transcribe_btn.click(
        gradio_transcribe,
        inputs=[audio_in, model_dropdown, beams_slider, rep_slider],
        outputs=[transcription_out],
    )

    with gr.Accordion("⚖️ Licenses, Attributions & Fair Usage Disclaimer", open=False):
        gr.Markdown("""
### 📜 Licensing Framework
1. **Base Architecture**: Useful Sensors Moonshine ASR ([Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)).
2. **Fine-Tuning Datasets**: OpenBible, Google FLEURS, and NaijaVoices datasets licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.
3. **Usage Terms**: Free for research, cultural preservation, evaluation benchmarks, and non-commercial African speech recognition initiatives.

### 🛡️ Attribution & Citations
- Fine-tuned by [Samuel Olubukun](https://github.com/samolubukun).
- Datasets credited to **NaijaVoices Community** (*Emezue et al., 2025*), **Google FLEURS**, and **OpenBible Project**.
""")

    with gr.Accordion("🔌 OpenAI API Documentation & Examples", open=False):
        gr.Markdown("""
Use this Space as a direct drop-in replacement for OpenAI's Whisper API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR-SPACE-NAME>.hf.space/v1",
    api_key="your-key",  # If MOONSHINE_API_KEY secret is configured
)

with open("speech.wav", "rb") as f:
    res = client.audio.transcriptions.create(
        model="samuelolubukun/moonshine-tiny-hausa-v2",
        file=f,
    )
print(res.text)
```
""")

class ApiDelegateRoute(BaseRoute):
    def matches(self, scope):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if p == "/healthz" or p == "/v1" or p.startswith("/v1/"):
                return Match.FULL, {}
        return Match.NONE, {}

    async def handle(self, scope, receive, send):
        await fastapi_app(scope, receive, send)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, prevent_thread_lock=True)
    demo.app.router.routes.insert(0, ApiDelegateRoute())
    logger.info("OpenAI API routes (/healthz, /v1/*) attached to Gradio server.")
    demo.block_thread()
