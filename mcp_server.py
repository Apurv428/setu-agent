"""
mcp_server.py — MCP server "sarvam-tools" (FastMCP).

Wraps every Sarvam capability as an MCP tool so both agents (scratch and
LangGraph) can consume the same server without knowing the SDK internals.

Run in isolation:  mcp dev mcp_server.py
"""

from mcp.server.fastmcp import FastMCP
import sarvam_client as sc

mcp = FastMCP("sarvam-tools")


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
    """Search the local knowledge base for facts about Indian languages, Sarvam AI models, and Indic scripts. Call this before answer_question when the question is about Indian languages, their history, scripts, speaker counts, or Sarvam's capabilities."""
    import retrieval
    try:
        passages = retrieval.search(query, k=3)
    except Exception as exc:
        return f"Knowledge search unavailable: {exc}"
    if not passages:
        return "No relevant passages found."
    return "\n\n---\n\n".join(passages)


if __name__ == "__main__":
    mcp.run()
