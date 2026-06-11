"""
sarvam_client.py — single source of truth for all Sarvam API calls.

All other modules import from here; no other file calls Sarvam directly.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI as _OpenAI
from sarvamai import SarvamAI
from sarvamai.play import save as _save_audio

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

_client: SarvamAI | None = None
_openai_client: _OpenAI | None = None


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError("SARVAM_API_KEY not set — copy .env.example to .env and add your key")
        _client = SarvamAI(api_subscription_key=key)
    return _client


def _get_openai_client() -> _OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError("SARVAM_API_KEY not set")
        _openai_client = _OpenAI(api_key=key, base_url="https://api.sarvam.ai/v1")
    return _openai_client


@dataclass
class TranscribeResult:
    text: str
    language_code: str  # detected language, e.g. "hi-IN"


def transcribe(audio_path: str, mode: str = "transcribe") -> TranscribeResult:
    """
    Transcribe speech to text using Saaras v3.

    mode: transcribe | translate | verbatim | translit | codemix
    Returns the transcript text and the detected language code.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        resp = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode=mode,
        )
    text = resp.transcript
    lang = getattr(resp, "language_code", "unknown")
    return TranscribeResult(text=text, language_code=lang)


def translate(text: str, target_language_code: str, source_language_code: str = "auto") -> str:
    """
    Translate text using Sarvam-Translate / Mayura.

    source_language_code defaults to "auto" for automatic detection.
    target_language_code must be a BCP-47 code, e.g. "hi-IN".
    """
    client = _get_client()
    resp = client.text.translate(
        input=text,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )
    return resp.translated_text


def chat(messages: list[dict], model: str = "sarvam-30b") -> str:
    """
    Send a chat completion request to a Sarvam model.

    Uses the OpenAI-compatible endpoint at api.sarvam.ai/v1.
    messages: OpenAI-style list of {"role": ..., "content": ...} dicts.
    Returns the assistant's reply text.
    """
    import json as _json

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    message = resp.choices[0].message
    content = message.content

    # sarvam-30b may return native tool_calls with null content even when no tools
    # are registered; convert to the JSON protocol the scratch agent expects.
    if content is None:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            call = tool_calls[0]
            try:
                args = _json.loads(call.function.arguments)
            except Exception:
                args = {}
            return _json.dumps({"tool": call.function.name, "args": args})
        return ""

    return content


def synthesize(text: str, target_language_code: str, out_path: str = "reply.wav", speaker: str = "shubh") -> str:
    """
    Convert text to speech using Bulbul v3.

    Returns the path to the written WAV file.
    Text is capped at ~2500 chars by the API; truncate before calling if needed.
    """
    client = _get_client()
    resp = client.text_to_speech.convert(
        text=text,
        target_language_code=target_language_code,
        model="bulbul:v3",
        speaker=speaker,
    )
    _save_audio(resp, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Isolated smoke tests — run with:  python sarvam_client.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _banner(title: str) -> None:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

    # ── 1. translate ──────────────────────────────────────────────
    _banner("1 / 4  translate")
    result = translate("Hello, how are you?", target_language_code="hi-IN")
    print("Input : Hello, how are you?")
    print("Output:", result)
    assert isinstance(result, str) and len(result) > 0, "translate returned empty"
    print("PASS")

    # ── 2. chat ───────────────────────────────────────────────────
    _banner("2 / 4  chat  (sarvam-30b)")
    reply = chat([{"role": "user", "content": "What is the capital of India? Reply in one sentence."}])
    print("Answer:", reply)
    assert isinstance(reply, str) and len(reply) > 0, "chat returned empty"
    print("PASS")

    # ── 3. synthesize ─────────────────────────────────────────────
    _banner("3 / 4  synthesize  (Bulbul v3)")
    out = synthesize("नमस्ते, आप कैसे हैं?", target_language_code="hi-IN", out_path="test_synth.wav")
    assert Path(out).exists(), f"synthesize: file not found at {out}"
    print(f"WAV written to: {out}  ({Path(out).stat().st_size} bytes)")
    print("PASS")

    # ── 4. transcribe ─────────────────────────────────────────────
    _banner("4 / 4  transcribe  (Saaras v3)")
    sample = Path("test_synth.wav")  # reuse the file we just synthesized
    if not sample.exists():
        print("SKIP — no audio file available; run synthesize first or place a WAV in samples/")
        sys.exit(0)
    tr = transcribe(str(sample))
    print(f"Transcript  : {tr.text}")
    print(f"Detected lang: {tr.language_code}")
    assert isinstance(tr.text, str), "transcribe returned non-string"
    print("PASS")

    print("\nAll checks passed.")
