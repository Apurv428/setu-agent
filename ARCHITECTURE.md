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

Setu is intentionally small. One coherent repo demonstrates these capabilities at once:

1. **Hands-on use of Sarvam's models** — Saaras v3 (STT), Bulbul v3 (TTS),
   Sarvam-Translate / Mayura (translation), and sarvam-30b (reasoning).
2. **A shipped side project** — public on GitHub with a README and a demo.
3. **MCP servers** — we build our own MCP server that wraps the Sarvam tools.
4. **Authoring agents from scratch** — the agent loop is written twice: once with no
   framework, once with LangGraph, so the difference can be explained out loud.
5. **Retrieval-augmented answers** — a local embedding index over a knowledge base
   gives the agent a genuine reason to choose between tools.
6. **Eval pipeline** — a 14-case suite with LLM-as-judge scoring produces a real
   accuracy number that maps to production quality-measurement practice.
7. **Conversation memory** — both agents maintain context across turns within a session.

The single most important architectural idea is the **MCP boundary**: all tools live
behind one MCP server, and *both* agents consume that same server. That boundary is
what makes the difference between a script and a system.

---

## 2. System overview

```
  User speaks a question  (Hindi / Marathi / Tamil / ...)
        |  audio in
        v
  +-----------------------------+
  |  Voice entrypoint (app.py)  |   record a few seconds of audio,
  |  CLI, --chat REPL, --demo   |   play the spoken reply at the end
  +-----------------------------+
        |  audio file path / text
        v
  +-----------------------------+        tool calls (MCP protocol)
  |  Agent orchestrator         | ----------------------------------+
  |  - scratch_agent.py         |                                   |
  |    (no framework, history)  |                                   |
  |  - graph_agent.py           | <---------------------------------+
  |    (LangGraph + MemorySaver)|        tool results
  +-----------------------------+
                                                   |
                                                   v
                               +-------------------------------------+
                               |  MCP server: "sarvam-tools"         |
                               |  (mcp_server.py, FastMCP)           |
                               |  6 tools:                           |
                               |   transcribe_audio  --> Saaras v3   |
                               |   detect_language   --> sarvam-30b  |
                               |   translate_text    --> Sarvam-Translate
                               |   answer_question   --> sarvam-30b  |
                               |   synthesize_speech --> Bulbul v3   |
                               |   search_knowledge  --> retrieval.py|
                               +-------------------------------------+
                                    |                     |
                                    v                     v
                          +-----------------+   +------------------+
                          | sarvam_client.py|   |  retrieval.py    |
                          | single source   |   |  local RAG index |
                          | for Sarvam calls|   |  knowledge/*.md  |
                          +-----------------+   +------------------+
```

The agent **decides** which tools to call and in what order; it is not a hard-coded
pipeline. A typical turn for a knowledge question is: call `search_knowledge` first,
read the retrieved passages, then either call `answer_question` (if more reasoning is
needed) or answer directly from the passages. A typical voice turn adds
`transcribe_audio`, optional translation hops, and `synthesize_speech` at the end.

---

## 3. Layers, from the inside out

### 3.1 `sarvam_client.py` — the Sarvam wrapper

A thin module that is the **only** place in the codebase that knows Sarvam's request
shapes, model IDs, and response formats. Everything else calls these four functions.
If Sarvam changes a field name, exactly one file changes.

It wraps the official `sarvamai` Python SDK. Raw REST against the API is kept as a
documented fallback for debugging only.

Confirmed surface (June 2026 — re-verify against `docs.sarvam.ai` before relying on it):

| Function | Sarvam SDK call | Model / endpoint |
| --- | --- | --- |
| `transcribe(audio_path, ...)` | `client.speech_to_text.transcribe(file=..., model="saaras:v3", mode="transcribe")` | Saaras v3. Response carries the **detected language code**. |
| `translate(text, src, tgt)` | `client.text.translate(input=..., source_language_code="auto", target_language_code="hi-IN")` | Sarvam-Translate / Mayura. `"auto"` auto-detects source. |
| `chat(messages)` | `client.chat.completions.create(model="sarvam-30b", messages=[...])` | OpenAI-compatible; `base_url="https://api.sarvam.ai/v1"`. |
| `synthesize(text, tgt, out_path)` | `client.text_to_speech.convert(text=..., target_language_code="hi-IN", model="bulbul:v3", speaker="shubh")` | Bulbul v3. `from sarvamai.play import save` writes the WAV. |

### 3.2 `retrieval.py` — the local RAG layer

A lightweight retrieval module that sits between the MCP server and the `knowledge/`
directory. It is not an agent library — it is a single function the `search_knowledge`
tool calls.

**Knowledge base.** Six markdown documents (~150-250 words each) covering Indian
languages and scripts: an overview of the 22 scheduled languages, deep dives into
Hindi and Tamil, a survey of Indic scripts, a guide to India's four language families,
and a note on Sarvam AI's model lineup. This theme was chosen deliberately: these are
questions the chat model alone might answer vaguely, but retrieval answers precisely,
making `search_knowledge` meaningfully different from `answer_question`.

**Embedding model.** `paraphrase-multilingual-MiniLM-L12-v2` from sentence-transformers,
downloaded once and run locally with no API key. The multilingual model handles Hindi,
Marathi, and Tamil queries against English documents without a translation step.

**Index.** Plain numpy dot product over L2-normalized embeddings (equivalent to cosine
similarity). No vector database, no external dependency beyond numpy and
sentence-transformers.

**Chunking.** Each document is split on blank lines (paragraphs) and accumulated until
a chunk reaches ~120 words. Each chunk carries its source filename.

**Cache.** On first call, embeddings are built and saved to `knowledge/.index.npz`.
On subsequent calls, the cache is loaded if it is newer than all documents; otherwise
it is rebuilt. This keeps startup fast after the first run.

**Public API:**
```python
search(query: str, k: int = 3) -> list[dict]
# each dict: {"source": str, "text": str, "score": float}
```

**Standalone use:**
```bash
python retrieval.py "Devanagari"
```

### 3.3 `mcp_server.py` — the MCP server ("sarvam-tools")

Built with **FastMCP**. It exposes each capability as an MCP tool; the docstring is
the agent-facing interface, so it is part of the contract rather than a comment.

Six tools:

- `transcribe_audio(audio_path) -> str` — Saaras v3 STT; returns transcript + detected language
- `detect_language(text) -> str` — sarvam-30b LID; returns a BCP-47 code
- `translate_text(text, source_language_code, target_language_code) -> str` — Sarvam-Translate
- `answer_question(question) -> str` — sarvam-30b chat
- `synthesize_speech(text, target_language_code) -> str` — Bulbul v3; returns the WAV path
- `search_knowledge(query) -> str` — calls `retrieval.search`; returns formatted passages,
  or `"NO_RELEVANT_KNOWLEDGE_FOUND"` when all scores fall below 0.35

The docstring for `search_knowledge` instructs the agent to call it *before*
`answer_question` for questions about Indian languages and scripts, and to fall back
to `answer_question` on the sentinel string. This turns score-based retrieval
confidence into a tool-choice signal the agent can act on without any code change.

The server also exposes two resources (`sarvam://languages`, `sarvam://models`) and
three prompt templates (`answer_in_language`, `translate_prompt`, `voice_agent_turn`).

The server runs over **stdio** transport and is verified in isolation with `mcp dev`
before any agent is wired to it.

### 3.4 The agents

Both agents talk to the *same* `sarvam-tools` MCP server. Neither agent owns the
tools; the server is a clean, independently-testable boundary.

**`scratch_agent.py` — the differentiator.** A tool-calling loop with no orchestration
framework. The core loop is: the model proposes an action as JSON, we parse it, execute
the named tool, append the observation, and repeat until the model emits a final answer
or we hit the step cap. The protocol is a strict JSON contract:

```
{"tool": "<name>", "args": {...}}   -> call a tool
{"final": "<answer>"}               -> done
```

TOOLS dict maps every tool name to a local function that calls the appropriate
`sarvam_client` or `retrieval` function. The system prompt tells the model when to use
`search_knowledge` (before `answer_question` for language/script questions) and what
to do with `NO_RELEVANT_KNOWLEDGE_FOUND`.

Multi-turn memory works by passing the accumulated `messages` list back into `run()`
on the next call. The caller (usually `app.py`) owns the history object.

**`graph_agent.py` — the framework version.** The same behaviour built with LangGraph.
It connects to the MCP server through `langchain-mcp-adapters`, pulls all six tools
automatically (including `search_knowledge`), and hands them to a prebuilt ReAct agent.

```python
llm = ChatOpenAI(model="sarvam-30b",
                 base_url="https://api.sarvam.ai/v1",
                 api_key=SARVAM_API_KEY)

client = MultiServerMCPClient({
    "sarvam": {"command": "python", "args": ["mcp_server.py"], "transport": "stdio"}
})
tools = await client.get_tools()
agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
```

`MemorySaver` stores the full message graph keyed by `thread_id`. Reusing the same
`thread_id` across calls gives the graph agent conversation memory equivalent to
passing `history` in the scratch agent.

### 3.5 `app.py` — the voice entrypoint

A thin CLI with three modes:

- **Default** — record mic for N seconds, run the chosen agent, play the WAV reply.
- **`--text`** — skip the mic, pass a text string directly.
- **`--chat`** — multi-turn REPL that keeps context until the user types `quit`.
  Uses `history` (scratch agent) or a fixed `thread_id` (graph agent).
- **`--demo`** — three scripted turns with no mic, useful for recording demos.

---

## 4. Data flow of one turn (the agent's decision, not a fixed pipeline)

1. `app.py` captures audio and passes the file path (or a text string) to the agent.
2. For a knowledge question, the agent calls `search_knowledge` first. If relevant
   passages are returned, it may answer directly from them or use them to augment
   `answer_question`. If `NO_RELEVANT_KNOWLEDGE_FOUND` is returned, it falls back to
   `answer_question` immediately.
3. For a voice input, the agent calls `transcribe_audio` to get the question text and
   the detected language. It may then call `translate_text` to English before reasoning,
   then `translate_text` back to the user's language, or handle the language natively.
4. The agent calls `synthesize_speech` with the final answer and the target language.
5. `app.py` plays the returned WAV.

The sequencing is always the agent's call. Step 2 and step 3 can be interleaved; the
agent may skip translation if the model handles the language directly; it may call
`search_knowledge` after transcription once it knows the question topic. Different
questions produce different tool sequences — that is the whole point.

---

## 5. Eval pipeline

The eval suite (`eval/`) provides a concrete quality measurement, independent of
manual inspection.

**Dataset** (`eval/dataset.json`). 14 labeled cases across four categories:

| Category | Cases | Scoring |
| --- | --- | --- |
| `qa` | 4 | Run through scratch agent (exercises RAG), scored by LLM-as-judge |
| `translation` | 4 | `sc.translate`, scored by LLM-as-judge |
| `lang_detect` | 3 | `sc.chat` for BCP-47 code, exact match |
| `round_trip` | 3 | TTS then STT, difflib similarity >= 0.80 |

**Judge.** The LLM-as-judge prompt demands strict JSON: `{"verdict": "PASS" or "FAIL",
"reason": "<one line>"}`. On malformed output, the runner re-asks once with a clearer
prompt. If the second reply is also unparseable, the case is counted as FAIL with
`"judge_parse_error"` recorded. API errors are also counted as FAIL with the error
message recorded — the runner never crashes mid-run.

**`qa` cases route through the scratch agent** (not directly through `sc.chat`) so the
eval measures the full RAG + reasoning path, not just the raw model.

**Results (run 2026-06-11):**

| Category | Passed | Total | Accuracy |
| --- | --- | --- | --- |
| qa | 3 | 4 | 75.0% |
| translation | 4 | 4 | 100.0% |
| lang_detect | 3 | 3 | 100.0% |
| round_trip | 3 | 3 | 100.0% |
| **Overall** | **13** | **14** | **92.9%** |

---

## 6. Tests

`pytest -q` runs 42 tests in about 1 second with zero network calls and no model
downloads. All Sarvam API calls and the embedding model are mocked.

| Test file | What it covers |
| --- | --- |
| `tests/test_retrieval.py` | Chunking produces non-empty chunks with sources; cosine top-k ordering is correct on a synthetic matrix; cache is stale when a doc is newer; cache is fresh when docs are older |
| `tests/test_tools.py` | Every MCP tool returns the expected shape with `sc` mocked; `search_knowledge` formats results correctly; returns `NO_RELEVANT_KNOWLEDGE_FOUND` on low/empty scores |
| `tests/test_scratch_agent.py` | Happy path (tool call then final); malformed JSON recovery; unknown tool reports available tools; max_steps stops the loop; history carries over across turns |
| `tests/test_eval_scoring.py` | Round-trip similarity PASS/FAIL at the 0.80 threshold; judge parses valid JSON; malformed-then-valid recovers; malformed-twice counts as FAIL; API error counts as FAIL |

---

## 7. Tech stack

- Python 3.11+
- `sarvamai` — official Sarvam SDK (primary); raw REST kept as a debug fallback
- `mcp` — the MCP Python SDK (FastMCP) for the server
- `sentence-transformers` — local embedding model (no API key)
- `numpy` — cosine similarity for the retrieval index
- `langgraph` + `langchain-mcp-adapters` + `langchain-openai` — the framework agent
- `sounddevice` + `scipy` — audio recording and playback
- `python-dotenv` — config from `.env`
- `pytest` — test suite

---

## 8. Key design decisions

**Why MCP instead of importing the functions directly.** MCP decouples the tools from
any single agent. The same server backs the scratch loop, the LangGraph agent, the MCP
inspector, and future clients — tested in isolation before any agent touches it.

**Why two agents.** To show what a framework does *for* you and what it *hides*. The
scratch loop makes the mechanism visible (you can read the prompt, the parse, the
dispatch). LangGraph hides those behind state management and graph structure. Building
the loop by hand earns the right to say you can author agents without a framework.

**Why `search_knowledge` as an MCP tool rather than a pre-retrieval step.** Making
retrieval a tool keeps the agent in charge. The agent reads the question, decides
whether it warrants a knowledge lookup, and then decides what to do with the result.
A pre-retrieval step would run on every turn and add latency even for arithmetic
questions. Encoding the "when to use it" guidance in the tool's docstring is how the
agent learns the policy without any code logic.

**Why a local embedding model.** No additional API key, no per-query cost, no network
round-trip for embedding. The multilingual MiniLM model handles Indic-script queries
against English documents without a translation step. The 10-second first-run cost is
amortised by the `.index.npz` cache.

**Why a single `sarvam_client.py`.** One file owns every Sarvam-specific detail, so
API drift has exactly one blast radius.

**Why the eval routes `qa` cases through the scratch agent.** The eval measures the
system, not the model. A `qa` case answered directly via `sc.chat` would skip the RAG
path and might pass or fail for different reasons than a real user interaction would.
Routing through the agent exercises the full `search_knowledge -> answer_question`
path and gives a number that reflects real production behaviour.

---

## 9. Failure modes the design accounts for

- **ASR misrecognition.** Saaras v3's `codemix` / `transcribe` modes help. Surface
  low-confidence transcripts rather than hiding them.
- **Wrong language detection.** Prefer the language Saaras returns from transcription;
  fall back to `translate(auto)`.
- **Malformed JSON from the model in the scratch loop.** Guard the parse, re-prompt
  with a reminder of the contract, cap retries.
- **No relevant knowledge.** `search_knowledge` returns `NO_RELEVANT_KNOWLEDGE_FOUND`
  when all chunk scores fall below 0.35; the agent falls back to `answer_question`.
- **Stale RAG index.** The cache is invalidated on mtime: any document newer than
  `.index.npz` triggers a rebuild.
- **Malformed judge output in eval.** One re-ask, then count as FAIL. API errors also
  count as FAIL. The runner never crashes mid-run.
- **API latency / timeouts.** The loop's step cap prevents runaway tool calls.

---

## 10. Future directions

- **Streaming STT/TTS** over WebSocket for lower latency (both Saaras and Bulbul
  offer it; not wired up here).
- **Observability** — tool calls print to stdout but are not traced to any structured
  logging system. LangSmith or a simple spans table would make debugging easier at
  scale.
- **Guardrails** — no content filtering before TTS output. A lightweight classifier
  or a judge step before `synthesize_speech` would be needed for production.
