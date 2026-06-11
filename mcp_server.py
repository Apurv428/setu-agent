"""
mcp_server.py — MCP server "sarvam-tools" (FastMCP).

Wraps the five core Sarvam capabilities as MCP tools so both agents
(scratch and LangGraph) can consume the same server.

  stdio (used by agents):      python mcp_server.py
  dev inspector:               .\run_inspector.ps1
  HTTP/SSE (remote clients):   python mcp_server.py --http
    → connect to http://localhost:8000/sse
"""

import sys
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from pydantic import Field

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from mcp.server.fastmcp import FastMCP
import sarvam_client as sc

mcp = FastMCP("sarvam-tools")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def transcribe_audio(audio_path: str) -> str:
    """Transcribe an Indian-language audio file to text using Saaras v3; returns transcript and detected language."""
    result = sc.transcribe(audio_path)
    return f"Transcript: {result.text}\nLanguage: {result.language_code}"


@mcp.tool()
def detect_language(text: str) -> str:
    """Detect the language of a text string; returns a BCP-47 code such as hi-IN or ta-IN."""
    reply = sc.chat([{
        "role": "user",
        "content": (
            "Identify the language of the following text and reply with ONLY its BCP-47 code "
            "(e.g. hi-IN, ta-IN, mr-IN, en-IN). No explanation.\n\n" + text
        ),
    }])
    return reply.strip()


@mcp.tool()
def translate_text(text: str, source_language_code: str, target_language_code: str) -> str:
    """Translate text between Indic languages and English. Use 'auto' as source_language_code to auto-detect."""
    return sc.translate(
        text,
        target_language_code=target_language_code,
        source_language_code=source_language_code,
    )


@mcp.tool()
def answer_question(question: str) -> str:
    """Answer a question using the Sarvam chat model (sarvam-30b)."""
    return sc.chat([{"role": "user", "content": question}])


@mcp.tool()
def synthesize_speech(text: str, target_language_code: str) -> str:
    """Convert text to speech in the target language using Bulbul v3; returns the saved audio file path."""
    return sc.synthesize(text, target_language_code=target_language_code)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Search the local knowledge base about Indian languages and scripts. Use this BEFORE answer_question when the user asks about Indian languages, scripts, or related facts."""
    import retrieval
    LOW_SCORE = 0.35
    results = retrieval.search(query, k=3)
    if not results or all(r["score"] < LOW_SCORE for r in results):
        return "NO_RELEVANT_KNOWLEDGE_FOUND"
    parts = []
    for r in results:
        parts.append(f"[{r['source']} | score {r['score']:.2f}]\n{r['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Resources — static reference data exposed to any MCP client
# ---------------------------------------------------------------------------

@mcp.resource("sarvam://languages")
def supported_languages() -> str:
    """All BCP-47 language codes supported by Sarvam STT, TTS, and Translate."""
    return (
        "hi-IN  Hindi\n"
        "mr-IN  Marathi\n"
        "ta-IN  Tamil\n"
        "te-IN  Telugu\n"
        "bn-IN  Bengali\n"
        "gu-IN  Gujarati\n"
        "kn-IN  Kannada\n"
        "ml-IN  Malayalam\n"
        "pa-IN  Punjabi\n"
        "od-IN  Odia\n"
        "en-IN  English (Indian)\n"
    )


@mcp.resource("sarvam://models")
def available_models() -> str:
    """Sarvam AI models available through this server."""
    return (
        "STT   saaras:v3     — speech-to-text, 11 Indic languages\n"
        "TTS   bulbul:v3     — text-to-speech, speaker: shubh (default)\n"
        "Chat  sarvam-30b    — 64K context, native tool calling (recommended)\n"
        "Chat  sarvam-105b   — 128K context, flagship\n"
        "MT    mayura         — Sarvam-Translate, source can be 'auto'\n"
    )


# ---------------------------------------------------------------------------
# Prompts — reusable templates an agent or user can invoke by name
# ---------------------------------------------------------------------------

@mcp.prompt()
def answer_in_language(
    question: Annotated[str, Field(description="The question to answer")],
    language_code: Annotated[str, Field(description="BCP-47 code — e.g. hi-IN, ta-IN, mr-IN, te-IN, bn-IN")],
) -> list[dict]:
    """Answer a question and reply entirely in the specified Indic language."""
    return [{
        "role": "user",
        "content": (
            f"Answer the following question clearly and concisely. "
            f"Your entire response must be in the language with BCP-47 code '{language_code}'.\n\n"
            f"Question: {question}"
        ),
    }]


@mcp.prompt()
def translate_prompt(
    text: Annotated[str, Field(description="Text to translate")],
    target_language_code: Annotated[str, Field(description="Target BCP-47 code — e.g. hi-IN, ta-IN")],
    source_language_code: Annotated[str, Field(description="Source BCP-47 code, or 'auto' to detect")],
) -> list[dict]:
    """Translate text between languages using Sarvam Translate."""
    return [{
        "role": "user",
        "content": (
            f"Translate the text below from '{source_language_code}' to '{target_language_code}'. "
            f"Return only the translated text, no explanation.\n\n{text}"
        ),
    }]


@mcp.prompt()
def voice_agent_turn(
    user_utterance: Annotated[str, Field(description="Transcribed text of what the user said")],
    detected_language: Annotated[str, Field(description="BCP-47 code of the user's spoken language — e.g. hi-IN")],
) -> list[dict]:
    """Full voice-agent turn: transcription → reasoning → synthesised reply."""
    return [{
        "role": "user",
        "content": (
            f"You are Setu, a multilingual voice assistant powered by Sarvam AI.\n"
            f"The user spoke in '{detected_language}'. "
            f"Their message (transcribed): {user_utterance}\n\n"
            f"Steps:\n"
            f"1. Translate the utterance to English if needed.\n"
            f"2. Answer the question accurately.\n"
            f"3. Translate your answer back to '{detected_language}'.\n"
            f"4. Call synthesize_speech with the answer and target_language_code='{detected_language}'.\n"
            f"5. Return the answer text and the audio file path."
        ),
    }]


if __name__ == "__main__":
    if "--http" in sys.argv:
        print("sarvam-tools MCP server → http://localhost:8000/sse", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        mcp.run()
