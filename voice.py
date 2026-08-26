"""
Voice I/O for the "voice-first" requirement in the pitch deck.

STT: faster-whisper — runs locally, no API key needed, handles Hindi + English
     (and code-switched speech, common in real Indian usage) reasonably well.
TTS: edge-tts — free, no API key, good multilingual voices including Hindi.

Both are wrapped behind simple functions so they can be swapped for a paid cloud
provider (e.g. Sarvam AI, Google, Azure) later without touching calling code.
"""
import os
import uuid
from functools import lru_cache

import edge_tts
from faster_whisper import WhisperModel

from app.config import settings

os.makedirs(settings.AUDIO_STORAGE_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    # int8 compute type keeps this usable on CPU-only servers; switch to "float16" if on GPU.
    return WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_audio(file_bytes: bytes, filename_hint: str = "audio.wav") -> tuple[str, str]:
    """Returns (transcript, detected_language_code)."""
    tmp_path = os.path.join(settings.AUDIO_STORAGE_DIR, f"in_{uuid.uuid4()}_{filename_hint}")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments)
        detected_language = info.language or "en"
        return transcript.strip(), detected_language
    finally:
        os.remove(tmp_path)


async def synthesize_speech(text: str, language: str = "hi") -> str:
    """Generates speech audio for `text`, saves it to disk, and returns a public URL."""
    voice = settings.TTS_VOICE_HI if language == "hi" else settings.TTS_VOICE_EN
    filename = f"reply_{uuid.uuid4()}.mp3"
    output_path = os.path.join(settings.AUDIO_STORAGE_DIR, filename)

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

    return f"{settings.MEDIA_BASE_URL}/{filename}"
