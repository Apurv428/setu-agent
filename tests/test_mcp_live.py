"""
tests/test_mcp_live.py — Live integration test for the sarvam-tools MCP server.

Connects to mcp_server.py via the actual MCP stdio transport and calls each
of the five core tools with real inputs.  Requires SARVAM_API_KEY in .env.

Usage:
  cd "d:\\Sarvam - Copy"
  .venv\\Scripts\\activate
  python tests/test_mcp_live.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

load_dotenv(Path(__file__).parent.parent / ".env")

# Run mcp_server.py from the project root so relative paths work correctly
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent.parent / "mcp_server.py")],
    env=None,
)

PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"


def _banner(title: str) -> None:
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


async def main() -> int:
    failures = 0

    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            print(f"Connected to sarvam-tools.  Tools: {tool_names}")
            assert set(tool_names) == {
                "transcribe_audio", "detect_language",
                "translate_text", "answer_question", "synthesize_speech",
            }, f"Unexpected tool list: {tool_names}"

            # ── 1. answer_question ────────────────────────────────────
            _banner("1 / 5  answer_question")
            r = await session.call_tool("answer_question", {
                "question": "What is the capital of India? One sentence only."
            })
            text = r.content[0].text
            print(f"  Response: {text[:200]}")
            ok = bool(text.strip())
            print(PASS if ok else FAIL)
            failures += 0 if ok else 1

            # ── 2. translate_text ─────────────────────────────────────
            _banner("2 / 5  translate_text")
            r = await session.call_tool("translate_text", {
                "text": "Hello, how are you?",
                "source_language_code": "en-IN",
                "target_language_code": "hi-IN",
            })
            text = r.content[0].text
            print(f"  Input : Hello, how are you?")
            print(f"  Output: {text}")
            ok = bool(text.strip())
            print(PASS if ok else FAIL)
            failures += 0 if ok else 1

            # ── 3. detect_language ────────────────────────────────────
            _banner("3 / 5  detect_language")
            sample = "नमस्ते, आप कैसे हैं?"
            r = await session.call_tool("detect_language", {"text": sample})
            text = r.content[0].text.strip()
            print(f"  Input    : {sample}")
            print(f"  Detected : {text}")
            ok = "hi" in text.lower()
            print(PASS if ok else FAIL)
            failures += 0 if ok else 1

            # ── 4. synthesize_speech ──────────────────────────────────
            _banner("4 / 5  synthesize_speech")
            r = await session.call_tool("synthesize_speech", {
                "text": "नमस्ते, मैं सेतु हूं।",
                "target_language_code": "hi-IN",
            })
            wav_path = r.content[0].text.strip()
            print(f"  Output file: {wav_path}")
            ok = Path(wav_path).exists()
            print(PASS if ok else FAIL + f" (file not found: {wav_path})")
            failures += 0 if ok else 1

            # ── 5. transcribe_audio ───────────────────────────────────
            _banner("5 / 5  transcribe_audio")
            # Use the WAV we just synthesised, or the existing smoke-test file
            wav = Path(wav_path) if ok else Path("test_synth.wav")
            if not wav.exists():
                print(f"  {SKIP} — no audio file; run: python sarvam_client.py first")
            else:
                r = await session.call_tool("transcribe_audio", {"audio_path": str(wav)})
                text = r.content[0].text
                print(f"  Result: {text}")
                ok2 = "Transcript:" in text
                print(PASS if ok2 else FAIL)
                failures += 0 if ok2 else 1

    print(f"\n{'='*55}")
    if failures == 0:
        print("  All tools passed.")
    else:
        print(f"  {failures} tool(s) FAILED.")
    print("=" * 55)
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
