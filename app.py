"""
app.py — CLI voice entrypoint for Setu.

Records audio from the microphone, runs the chosen agent, and plays
the spoken reply back.

Usage:
  python app.py                         # mic → scratch agent → speaker
  python app.py --agent graph           # mic → LangGraph agent → speaker
  python app.py --text "..."            # text query (skip recording)
  python app.py --seconds 8             # record longer (default 5 s)
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE   = 16_000
DEFAULT_SECS  = 5


def _record(out_path: str, seconds: int) -> str:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile

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


def _run_scratch(user_input: str) -> dict:
    from scratch_agent import run
    return run(user_input)


def _run_graph(user_input: str) -> str:
    from graph_agent import run
    return asyncio.run(run(user_input))


def main() -> None:
    parser = argparse.ArgumentParser(description="Setu — multilingual voice agent")
    parser.add_argument("--agent",   choices=["scratch", "graph"], default="scratch",
                        help="Which agent to use (default: scratch)")
    parser.add_argument("--text",    type=str, default=None,
                        help="Skip mic recording and use this text as input")
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECS,
                        help="Recording duration in seconds (default: 5)")
    args = parser.parse_args()

    # ── input ─────────────────────────────────────────────────────────────
    if args.text:
        user_input = args.text
        print(f"Text input: {user_input}")
    else:
        wav_in = os.path.join(tempfile.gettempdir(), "setu_input.wav")
        _record(wav_in, args.seconds)
        user_input = wav_in

    # ── agent ─────────────────────────────────────────────────────────────
    print(f"\nRunning {args.agent} agent...\n")

    if args.agent == "scratch":
        result    = _run_scratch(user_input)
        final     = result["final"]
        audio_out = result.get("audio_path")
    else:
        final     = _run_graph(user_input)
        audio_out = None

    print(f"\nAnswer: {final}")

    # ── playback ───────────────────────────────────────────────────────────
    if audio_out and Path(audio_out).exists():
        print("Playing spoken reply...")
        _play(audio_out)
    else:
        print("(No audio output — the agent did not call synthesize_speech)")


if __name__ == "__main__":
    main()
