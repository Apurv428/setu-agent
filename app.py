"""
app.py — CLI voice entrypoint for Setu.

Records audio from the microphone, runs the chosen agent, and plays
the spoken reply back.

Usage:
  python app.py                         # mic -> scratch agent -> speaker
  python app.py --agent graph           # mic -> LangGraph agent -> speaker
  python app.py --text "..."            # text query (skip recording)
  python app.py --seconds 8             # record longer (default 5 s)
  python app.py --chat                  # multi-turn chat REPL with memory
  python app.py --demo                  # scripted 3-turn demo, no mic needed
"""

import argparse
import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE  = 16_000
DEFAULT_SECS = 5

DEMO_TURNS = [
    "What is the most widely spoken language in India?",
    "How do you say 'good morning' in that language?",
    "Now translate that phrase into Tamil.",
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _record(out_path: str, seconds: int) -> str:
    import sounddevice as sd
    from scipy.io import wavfile
    import numpy as np

    print(f"  Recording {seconds}s — speak now...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    wavfile.write(out_path, SAMPLE_RATE, audio)
    print(f"  Saved to {out_path}")
    return out_path


def _play(path: str) -> None:
    import sounddevice as sd
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    sd.play(data, rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _run_scratch(
    user_input: str,
    history: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    from scratch_agent import run
    return run(user_input, history=history)


def _run_graph(user_input: str, thread_id: str = "default") -> str:
    from graph_agent import run
    return asyncio.run(run(user_input, thread_id=thread_id))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _single_turn(args: argparse.Namespace) -> None:
    """Original single-turn behaviour."""
    if args.text:
        user_input = args.text
        print(f"Text input: {user_input}")
    else:
        wav_in = os.path.join(tempfile.gettempdir(), "setu_input.wav")
        _record(wav_in, args.seconds)
        user_input = wav_in

    print(f"\nRunning {args.agent} agent...\n")

    if args.agent == "scratch":
        result, _history = _run_scratch(user_input)
        final     = result["final"]
        audio_out = result.get("audio_path")
    else:
        final     = _run_graph(user_input)
        audio_out = None

    print(f"\nAnswer: {final}")

    if audio_out and Path(audio_out).exists():
        print("Playing spoken reply...")
        _play(audio_out)
    else:
        print("(No audio output — the agent did not call synthesize_speech)")


def _chat_mode(agent: str) -> None:
    """Multi-turn REPL with conversation memory."""
    session_id = str(uuid.uuid4())
    history: list[dict] | None = None
    print(f"Setu ({agent} agent) — type 'quit' to exit\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if agent == "scratch":
            result, history = _run_scratch(text, history=history)
            reply     = result["final"]
            audio_out = result.get("audio_path")
        else:
            reply     = _run_graph(text, thread_id=session_id)
            audio_out = None

        print(f"Setu: {reply}\n")
        if audio_out and Path(audio_out).exists():
            _play(audio_out)


def _demo_mode() -> None:
    """Scripted 3-turn demo using the scratch agent. No mic required."""
    print("Setu demo — 3 scripted turns, scratch agent\n")
    history: list[dict] | None = None

    for i, question in enumerate(DEMO_TURNS, 1):
        print(f"\n{'='*55}")
        print(f"Turn {i}: {question}")
        print("=" * 55)
        result, history = _run_scratch(question, history=history)
        print(f"\nSetu: {result['final']}")
        if result.get("audio_path") and Path(result["audio_path"]).exists():
            print(f"Audio saved: {result['audio_path']}")

    print("\nDemo complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Setu — multilingual voice agent")
    parser.add_argument("--agent",   choices=["scratch", "graph"], default="scratch",
                        help="Which agent to use (default: scratch)")
    parser.add_argument("--text",    type=str, default=None,
                        help="Skip mic recording and use this text as input")
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECS,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--chat",    action="store_true",
                        help="Multi-turn chat mode with conversation memory")
    parser.add_argument("--demo",    action="store_true",
                        help="Run a scripted 3-turn demo without a microphone")
    args = parser.parse_args()

    if args.demo:
        _demo_mode()
    elif args.chat:
        _chat_mode(args.agent)
    else:
        _single_turn(args)


if __name__ == "__main__":
    main()
