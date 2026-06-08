# ARCHITECTURE — Setu

> Setu ("bridge") is a multilingual voice agent. You speak a question in any major
> Indian language; an AI agent reasons about it and speaks the answer back in the
> same language. Every Sarvam capability it uses (speech-to-text, translation,
> chat, text-to-speech) is exposed to the agent as a tool over an MCP server we
> build ourselves.

This document is the authoritative description of *how the system is shaped and why*.
For day-to-day build instructions and conventions, see `CLAUDE.md`. For what is
built so far and what to do next, see `STATE.md`.

---

## 1. What this project is, and why it is shaped this way

Setu is intentionally small. One coherent repo is meant to demonstrate four distinct
capabilities at once:

1. **Hands-on use of Sarvam's models** — Saaras v3 (STT), Bulbul v3 (TTS),
   Sarvam-Translate / Mayura (translation), and a Sarvam chat model (reasoning).
2. **A shipped side project** — public on GitHub with a README and a demo.
3. **MCP servers** — we build our own MCP server that wraps the Sarvam tools.
4. **Authoring agents from scratch** — the agent loop is written twice: once with no
   framework, once with LangGraph, so the difference can be explained out loud.

The single most important architectural idea is the **MCP boundary**: the Sarvam
tools live behind one MCP server, and *both* agents consume that same server. That
boundary is what makes the difference between a script and a system.

---

## 2. System overview

```
  User speaks a question  (Hindi / Marathi / Tamil / ...)
        |  audio in
        v
  +-----------------------------+
  |  Voice entrypoint (app.py)  |   record a few seconds of audio,
  |  CLI or Streamlit           |   play the spoken reply at the end
  +-----------------------------+
        |  audio file path / text
        v
  +-----------------------------+        tool calls (MCP protocol)
  |  Agent orchestrator         | ----------------------------------+
  |  - scratch_agent.py         |                                   |
  |    (no framework)           |                                   |
  |  - graph_agent.py           | <---------------------------------+
  |    (LangGraph)              |        tool results
  +-----------------------------+
        |                                          |
        | final answer text + target language      v
        |                          +-------------------------------------+
        |                          |  MCP server: "sarvam-tools"         |
        |                          |  (mcp_server.py, FastMCP)           |
        |                          |  tools:                             |
        |                          |   transcribe_audio   --> Saaras v3  |
        |                          |   detect_language     --> Sarvam LID |
        |                          |   translate_text      --> Sarvam-Translate / Mayura
        |                          |   answer_question     --> sarvam-30b (chat)
        |                          |   synthesize_speech   --> Bulbul v3 |
        |                          +-------------------------------------+
        |                                          |
        |                                          v
        |                          +-------------------------------------+
        |                          |  sarvam_client.py                   |
        |                          |  thin wrapper, single source of     |
        |                          |  truth for every Sarvam call        |
        |                          +-------------------------------------+
        v
  Spoken reply, in the user's own language
```

The agent **decides** which tools to call and in what order; it is not a hard-coded
pipeline. A typical run is: transcribe the audio, note the detected language, reason
about the question (translating to and from English only if that helps the LLM), then
synthesize the answer as speech in the user's language. The fact that the model owns
that sequencing decision is what makes this an agent rather than a function chain.

---

## 3. Layers, from the inside out

### 3.1 `sarvam_client.py` — the Sarvam wrapper

A thin module that is the **only** place in the codebase that knows Sarvam's request
shapes, model IDs, and response formats. Everything else calls these four or five
functions. If Sarvam changes a field name, exactly one file changes.

It wraps the official `sarvamai` Python SDK (preferred over raw REST because the SDK
is the path Sarvam documents and maintains, handles base64 audio decoding, and is far
less brittle than hand-rolled request bodies). Raw `requests` against the REST API is
kept as a documented fallback for debugging, not as the primary path.

Confirmed surface (June 2026 — re-verify against `docs.sarvam.ai` before relying on it):

| Function | Sarvam SDK call | Model / endpoint |
| --- | --- | --- |
| `transcribe(audio_path, ...)` | `client.speech_to_text.transcribe(file=..., model="saaras:v3", mode="transcribe")` | Saaras v3. Response also carries the **detected language code**. |
| `translate(text, src, tgt)` | `client.text.translate(input=..., source_language_code="auto", target_language_code="hi-IN")` | Sarvam-Translate / Mayura. `source_language_code="auto"` auto-detects. |
| `chat(prompt)` | `client.chat.completions.create(model="sarvam-30b", messages=[...])` | OpenAI-compatible; `base_url="https://api.sarvam.ai/v1"`. |
| `synthesize(text, tgt, out_path)` | `client.text_to_speech.convert(text=..., target_language_code="hi-IN", model="bulbul:v3", speaker="shubh")` | Bulbul v3. Audio is base64 in `.audios[0]`; `from sarvamai.play import save` writes a file. |

### 3.2 `mcp_server.py` — the MCP server ("sarvam-tools")

Built with **FastMCP** (`from mcp.server.fastmcp import FastMCP`). It exposes each
Sarvam capability as an MCP tool with a clear name, typed signature, and one-line
docstring (the docstring is what the agent's LLM reads to decide when to call it, so
it is part of the interface, not a comment). Tools:

- `transcribe_audio(audio_path, language_code="unknown") -> str`
- `detect_language(text) -> str` — maps to Sarvam's Language Identification API.
  Note: STT already returns a detected language, and `translate` accepts `"auto"`,
  so this tool is partly a convenience; confirm the exact LID method name in the docs,
  or fold it into the transcribe result if you want one fewer moving part.
- `translate_text(text, source_language_code, target_language_code) -> str`
- `answer_question(question) -> str`
- `synthesize_speech(text, target_language_code) -> str` (returns the saved audio path)

The server runs over **stdio** transport (`mcp.run()`), which is the right choice for a
local single-user tool. It is verified in isolation with the MCP inspector
(`mcp dev mcp_server.py`) *before* any agent is wired to it.

### 3.3 The agents

Both agents talk to the *same* `sarvam-tools` MCP server. That is the elegant part:
the server is a clean, reusable boundary that neither agent owns.

**`scratch_agent.py` — the differentiator.** A tool-calling loop with no orchestration
framework. The core loop is: the model proposes an action as JSON, we parse it, execute
the named tool, append the observation to the transcript, and repeat until the model
emits a final answer (or we hit a step cap). It is deliberately small and readable
because it is the artifact most likely to be walked through line-by-line in an
interview. The protocol is a strict JSON contract:

```
{"tool": "<name>", "args": {...}}   -> call a tool
{"final": "<answer>"}               -> done
```

Keeping it on a JSON protocol (rather than the model's native tool-calling) makes the
loop model-agnostic and makes the mechanism visible. The one real-world wrinkle to
handle is malformed JSON from the model — guard the parse and re-prompt rather than
crashing.

**`graph_agent.py` — the framework version.** The same behaviour built with LangGraph.
It connects to the MCP server through `langchain-mcp-adapters`, pulls the tools, and
hands them to a prebuilt ReAct agent. The LLM is Sarvam's chat model reached through
its OpenAI-compatible endpoint:

```python
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent   # or langchain.agents.create_agent

llm = ChatOpenAI(model="sarvam-30b",
                 base_url="https://api.sarvam.ai/v1",
                 api_key=SARVAM_API_KEY)

client = MultiServerMCPClient({
    "sarvam": {"command": "python", "args": ["mcp_server.py"], "transport": "stdio"}
})
tools = await client.get_tools()
agent = create_react_agent(llm, tools)
```

`sarvam-30b` supports native tool calling, which is what makes the LangGraph ReAct loop
work without a hand-written JSON protocol. That is precisely the contrast worth naming:
the framework version leans on the model's native tool-calling and LangGraph's state
machine; the scratch version makes both explicit.

### 3.4 `app.py` — the voice entrypoint

Deliberately thin. A CLI that records a few seconds with `sounddevice`, writes a WAV,
runs either agent, and plays back the returned audio — or a ~30-line Streamlit app with
an audio-recorder widget and a player. The agent is the substance; the UI exists so the
demo GIF looks real.

---

## 4. Data flow of one turn (the agent's decision, not a fixed pipeline)

1. `app.py` captures audio and passes the file path to the agent.
2. The agent calls `transcribe_audio` -> gets the question text and the detected language.
3. The agent reasons. If it judges that translating to English helps the LLM, it calls
   `translate_text` to English, calls `answer_question`, then `translate_text` back to
   the user's language. For a language the chat model handles well directly, it may skip
   the translation hops entirely. This branching is the agent's call.
4. The agent calls `synthesize_speech` with the final answer and the target language.
5. `app.py` plays the returned audio.

The order above is *typical*, not enforced. Different questions can produce different
tool sequences, which is the whole point.

---

## 5. Tech stack

- Python 3.11+
- `sarvamai` — official Sarvam SDK (primary); raw `requests` kept as a debug fallback
- `mcp` — the MCP Python SDK (FastMCP) for the server
- `langgraph` + `langchain-mcp-adapters` + `langchain-openai` — the framework agent
- `sounddevice` (CLI) or `streamlit` (UI) for the voice entrypoint
- `python-dotenv` for config; a free key from `dashboard.sarvam.ai`

Model IDs and field names shift; treat the table in §3.1 as a starting point and confirm
against `docs.sarvam.ai` before coding each call. This is logged in `STATE.md` so a
fresh session does not re-discover it.

---

## 6. Key design decisions

**Why MCP instead of importing the functions directly.** MCP decouples the Sarvam tools
from any single agent. The same server backs the scratch loop, the LangGraph agent, or
anything else later, and it can be tested in isolation with the inspector. That clean,
reusable server boundary is the explicit ARYA mandate ("MCP servers and agentic backend
infrastructure used across teams").

**Why two agents.** To show what a framework does *for* you and what it *hides*. The
scratch loop makes the mechanism visible (you can see the prompt, the parse, the
dispatch). LangGraph hides those behind state management, retries, and graph structure.
Building the loop once by hand is what earns the right to say you can author agents
without a framework.

**Why the SDK over raw REST.** The SDK is the path Sarvam documents and keeps current,
it decodes audio for you, and it shrinks the surface area for field-name drift. Raw REST
stays documented as a fallback because explicit request bodies are occasionally the
fastest way to debug a failing call.

**Why a single `sarvam_client.py`.** One file owns every Sarvam-specific detail, so API
drift has exactly one blast radius.

**Related work — Sarvam's own MCP server.** Sarvam now publishes a hosted MCP server.
We build our own anyway, because the point of the exercise is to demonstrate building
and owning the boundary, and because a self-built server is what you can test, version,
and explain in full. Knowing the hosted one exists is itself a good signal of awareness.

---

## 7. Failure modes the design accounts for

- **ASR misrecognition**, especially on code-mixed speech. Saaras v3's `codemix` /
  `transcribe` modes help; surface low-confidence transcripts rather than hiding them.
- **Wrong language detection.** Prefer the language Saaras returns from transcription;
  fall back to `translate`'s `auto` detection.
- **Malformed JSON from the model in the scratch loop.** Guard the parse, re-prompt
  with a reminder of the contract, and cap retries.
- **API latency / timeouts.** Set sensible timeouts in the wrapper; the loop's step cap
  prevents runaway tool calls.

---

## 8. Out of scope (for the weekend) and future directions

- Streaming STT/TTS over WebSocket for lower latency (both Saaras and Bulbul offer it).
- A `search_knowledge` tool backed by a small RAG index, giving the agent a real reason
  to choose between tools (and a legitimate RAG claim).
- An eval harness: 10–15 question/expected-answer pairs scored into one accuracy number,
  which maps directly to Sarvam's "set up eval pipelines and define quality metrics."
- Memory across turns, tool-call observability, and guardrails before TTS goes out.

These are explicitly deferred so the core — voice in, agent reasoning over MCP tools,
voice out, built two ways — actually ships in a weekend.