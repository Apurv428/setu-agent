# Setu — Multilingual Voice Agent on Sarvam AI

> *Setu* (सेतु) means **bridge**. Speak a question in any major Indian language; an AI agent reasons over Sarvam's speech, translation, and chat tools and speaks the answer back in your language.

**Live demo: [setu-agent.onrender.com](https://setu-agent.onrender.com)**

<br>

## Demo

Try the live web app at [setu-agent.onrender.com](https://setu-agent.onrender.com) — no setup needed.

Or run the scripted CLI demo without a microphone:

```bash
python app.py --demo
```

This runs 3 scripted turns (each a follow-up on the last, to show memory works),
prints every tool call the agent makes, and saves the spoken replies as WAV files.

*(add demo.gif here after recording)*

<br>

## What this project demonstrates

| Capability | How |
|---|---|
| Hands-on use of Sarvam models | Saaras v3 (STT), Bulbul v3 (TTS), Sarvam-Translate, sarvam-30b (chat) |
| Building an MCP server from scratch | FastMCP server with 6 tools, 2 resources, 3 prompts — testable in the MCP Inspector |
| Authoring an agent without a framework | `scratch_agent.py` — a hand-written JSON tool-call loop, no LangChain/LangGraph |
| Authoring the same agent with a framework | `graph_agent.py` — LangGraph ReAct consuming the same MCP server |
| Retrieval-augmented answers | `retrieval.py` + `search_knowledge` MCP tool — local embeddings over a knowledge base |
| Measuring quality | `eval/run_eval.py` — 14-case eval suite with LLM-as-judge scoring |
| Web interface + cloud deployment | `streamlit_app.py` deployed on Render with mic and text input |

<br>

## Architecture

```
  User (browser or CLI)
        │
        ├── streamlit_app.py  (web — text input + mic recording)
        │
        └── app.py            (CLI — mic, --chat REPL, --text, --demo)
                │
                │  audio path / text query
                ▼
  ┌──────────────────────────┐      tool calls over MCP / JSON protocol
  │  Agent orchestrator      │ ─────────────────────────────────────────┐
  │                          │                                          │
  │  scratch_agent.py        │                                          │
  │  (no framework)    OR    │ ◄────────────────────────────────────────┘
  │  graph_agent.py          │      tool results
  │  (LangGraph)             │
  └──────────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────┐
                               │  mcp_server.py               │
                               │  "sarvam-tools"  (FastMCP)   │
                               │                              │
                               │  transcribe_audio  ────────► Saaras v3
                               │  detect_language   ────────► sarvam-30b
                               │  translate_text    ────────► Sarvam-Translate
                               │  answer_question   ────────► sarvam-30b
                               │  synthesize_speech ────────► Bulbul v3
                               │  search_knowledge  ────────► retrieval.py
                               └──────────────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────┐
                               │  sarvam_client.py            │
                               │  single source of truth      │
                               │  for every Sarvam API call   │
                               └──────────────────────────────┘
```

The agent **decides** which tools to call and in what order. A typical turn looks like:

1. `search_knowledge` — check the local knowledge base first (for questions about Indian languages/scripts)
2. `transcribe_audio` — WAV → text + detected language (e.g. `hi-IN`)
3. `translate_text` — translate question to English for better reasoning accuracy
4. `answer_question` — get the answer from `sarvam-30b`
5. `translate_text` — translate answer back to the user's language
6. `synthesize_speech` — text → WAV via Bulbul v3

The agent may skip steps (e.g. answer directly in Hindi without translation hops when the model handles it natively). That decision is the agent's, not hard-coded logic.

<br>

## Tech stack

| Layer | Library / Model |
|---|---|
| Speech-to-text | Sarvam **Saaras v3** |
| Translation | Sarvam **Sarvam-Translate / Mayura** |
| Chat / reasoning | Sarvam **sarvam-30b** (64 K context, native tool calling) |
| Text-to-speech | Sarvam **Bulbul v3** |
| MCP server | **FastMCP** (`mcp` Python SDK) |
| Framework agent | **LangGraph** + `langchain-mcp-adapters` + `langchain-openai` |
| Embeddings (RAG) | `sentence-transformers` `paraphrase-multilingual-MiniLM-L12-v2` (local, no API) |
| Web interface | **Streamlit** — text input + `st.audio_input` mic widget |
| Audio I/O (CLI) | `sounddevice` + `scipy` |
| Deployment | **Render** (web service, auto-deploy from GitHub) |
| Config | `python-dotenv` |

<br>

## Project structure

```
setu/
├── sarvam_client.py      # Thin wrapper — only file that calls Sarvam APIs
├── mcp_server.py         # FastMCP server: 6 tools + 2 resources + 3 prompts
├── retrieval.py          # Local RAG: embedding index over knowledge/ docs
├── scratch_agent.py      # Agent loop with NO framework (the differentiator)
├── graph_agent.py        # Same agent built with LangGraph
├── app.py                # CLI voice entrypoint: mic → agent → speaker
├── streamlit_app.py      # Web interface: text + mic → agent → chat UI
├── render.yaml           # Render deployment config
├── run_inspector.ps1     # One-click MCP Inspector launcher (Windows)
├── knowledge/            # Markdown docs the agent can retrieve
│   ├── indian_languages_overview.md
│   ├── hindi_language.md
│   ├── tamil_language.md
│   ├── indic_scripts.md
│   ├── language_families.md
│   └── sarvam_ai.md
├── eval/
│   ├── dataset.json      # 14 labeled test cases
│   ├── run_eval.py       # Eval runner with LLM-as-judge
│   └── results.json      # Last run results
├── tests/
│   ├── test_retrieval.py     # Unit tests for retrieval.py (mocked, fast)
│   ├── test_tools.py         # Unit tests for MCP tools (mocked, fast)
│   ├── test_scratch_agent.py # Unit tests for the scratch agent loop
│   ├── test_eval_scoring.py  # Unit tests for judge/scoring logic
│   └── test_mcp_live.py      # Live integration test — real API calls via MCP stdio
├── requirements.txt
└── .env.example          # Copy to .env and add your key
```

<br>

## Quickstart

### 1. Get a free Sarvam API key

Sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) — it's free.

### 2. Clone and set up

```bash
git clone https://github.com/Apurv428/setu-agent.git
cd setu-agent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and set: SARVAM_API_KEY=your_key_here
```

### 3. Verify the Sarvam client

```bash
python sarvam_client.py
```

Expected: four `PASS` lines — translate → chat → synthesize → transcribe.

### 4. Run the Streamlit web app

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Type a question or use the mic tab to record. Each reply shows the tool call trace in a collapsible expander, and the synthesized audio plays inline.

### 5. Inspect the MCP server (Windows)

```powershell
.\run_inspector.ps1
```

Opens the MCP Inspector at `localhost:6274` with everything pre-configured. Click **Connect** — you'll see all 6 tools listed immediately. Select any tool, fill in the input, and click **Run Tool** to call the live API.

> On macOS/Linux: `mcp dev mcp_server.py` and add `SARVAM_API_KEY` in the Environment Variables panel.

### 6. Run the unit tests

```bash
pytest -q
```

42 tests, zero network calls, runs in about 3 seconds.

### 7. Run an agent from the CLI

```bash
# Framework-free scratch agent
python scratch_agent.py "Which script is Marathi written in?"

# LangGraph agent (same MCP server)
python graph_agent.py "Which script is Marathi written in?"
```

### 8. Full voice loop (CLI)

```bash
# Text input (no mic required)
python app.py --text "भारत की राजधानी क्या है?"

# Mic input — records 5 seconds
python app.py

# Multi-turn chat with memory (keeps context across follow-up questions)
python app.py --chat

# Scripted 3-turn demo, no mic required
python app.py --demo
```

### 9. Query the knowledge base directly

```bash
python retrieval.py "Devanagari"
python retrieval.py "Which languages use the same script as Hindi?"
```

Prints the top-3 relevant chunks with similarity scores. The index is built on first run and cached for fast subsequent queries.

### 10. Run the eval

```bash
python eval/run_eval.py
python eval/run_eval.py --category qa
python eval/run_eval.py --verbose
```

<br>

## Web interface

The Streamlit app (`streamlit_app.py`) is the recommended way to try Setu without setting up a local environment.

**Live:** [setu-agent.onrender.com](https://setu-agent.onrender.com)

**Two input modes:**
- **Text tab** — type any question, press Send
- **Voice tab** — click the mic, speak, click Stop — the agent runs automatically

**What you see per turn:**
- The assistant's text answer in a chat bubble
- A collapsible **Tool calls** expander showing every step the agent took (e.g. `step 1: search_knowledge(...)`)
- The synthesized Bulbul v3 audio playing inline below the answer

**Memory** is kept across turns within the browser session. Use the **Clear conversation** button in the sidebar to start fresh.

**Good questions to try:**
- "Which script is Marathi written in?" — triggers `search_knowledge` before answering
- "How many characters does the Tamil script have?" — retrieval + precise answer
- "Translate 'good morning' to Tamil"
- "What are the four language families of India?"
- Then follow up with "Which one has the most speakers?" — tests memory

<br>

## MCP Server — verified results

The server exposes **6 tools**, **2 resources**, and **3 prompt templates**, all verified live against the Sarvam API in the MCP Inspector.

---

### Tools

#### `detect_language`
Returns the BCP-47 language code of a text string.

```
Input : "kashi aahe"
Output: mr-IN          ← Marathi detected correctly

Input : "नमस्ते, आप कैसे हैं?"
Output: hi-IN

Input : "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
Output: ta-IN
```

#### `answer_question`
Answers a question using `sarvam-30b` (64K context, native tool calling).

```
Input : "What is the capital of Maharashtra?"
Output: "The capital of Maharashtra is Mumbai."

Input : "भारत की सबसे लंबी नदी कौन सी है?"
Output: "भारत की सबसे लंबी नदी गंगा है।"
```

#### `translate_text`
Translates between Indic languages and English. Pass `"auto"` as `source_language_code` to detect automatically.

```
Input : text="i am apurv"            source=en-IN   target=mr-IN
Output: "मी अपूर्व आहे"

Input : text="Good morning"          source=auto    target=mr-IN
Output: "शुभ सकाळ"
```

#### `synthesize_speech`
Converts text to speech using Bulbul v3. Returns the path to the saved WAV file.

#### `transcribe_audio`
Transcribes an Indian-language WAV file using Saaras v3. Returns transcript + detected language.

#### `search_knowledge`
Searches the local knowledge base (six markdown documents about Indian languages and scripts). Returns the top-3 relevant passages with source file and similarity score, or `NO_RELEVANT_KNOWLEDGE_FOUND` if no passage scores above 0.35.

```
Input : "Which script is Marathi written in?"
Output: [indic_scripts.md | score 0.70]
        Devanagari is the most widely used Indic script. It is used to write
        Hindi, Marathi, Sanskrit, Nepali, Konkani...
```

---

### Resources

| URI | Content |
|-----|---------|
| `sarvam://languages` | All 11 supported BCP-47 language codes with names |
| `sarvam://models` | Available Sarvam models (STT, TTS, Chat, Translate) with context limits |

---

### Prompts

| Prompt | Arguments | Use case |
|--------|-----------|----------|
| `answer_in_language` | `question`, `language_code` | Ask a question and get a reply in a specific Indic language |
| `translate_prompt` | `text`, `target_language_code`, `source_language_code` | Ready-to-use translation prompt with source/target |
| `voice_agent_turn` | `user_utterance`, `detected_language` | Full voice-agent turn: transcription → reasoning → synthesised reply |

<br>

## Live agent trace

Real output from `scratch_agent.py` on a Hinglish (code-mixed) query — the agent detects the language, answers, and replies in Hindi:

```
Query: Maharashtra ki rajdhani kya hai?

  step 1: detect_language({'text': 'Maharashtra ki rajdhani kya hai?'})
           -> hi-IN
  step 2: answer_question({'question': 'What is the capital of Maharashtra?'})
           -> The capital of Maharashtra is Mumbai. ...

Final: आप सही कह रहे हैं। महाराष्ट्र की आर्थिक राजधानी मुंबई है,
       जबकि नागपुर आधिकारिक राजधानी है।
```

<br>

## Retrieval-augmented answers

The `knowledge/` directory contains six markdown documents covering Indian languages and scripts: an overview of India's 22 scheduled languages, deep dives into Hindi and Tamil, a survey of Indic scripts (Devanagari, Tamil, Bengali, Telugu, Kannada, Malayalam, Gurmukhi), a guide to India's four language families, and a note on Sarvam AI's model lineup.

The agent uses `search_knowledge` automatically when a question is about Indian languages or scripts. The tool embeds the query using `paraphrase-multilingual-MiniLM-L12-v2` (a local sentence-transformers model that handles Hindi, Marathi, Tamil, and other Indic queries against English documents) and returns the top-3 passages by cosine similarity. If all scores fall below 0.35, it returns `NO_RELEVANT_KNOWLEDGE_FOUND` and the agent falls back to `answer_question`.

To verify retrieval standalone:

```bash
python retrieval.py "Devanagari"
python retrieval.py "Tamil classical language"
```

The embedding index is built on first run (~10 seconds) and cached to `knowledge/.index.npz`. It is rebuilt automatically if any document is newer than the cache file.

<br>

## Measuring quality

The eval suite lives in `eval/`. Run it with:

```bash
python eval/run_eval.py
```

### Categories

| Category | Cases | What it tests |
|---|---|---|
| `qa` | 4 | Questions answered via the scratch agent (exercises both RAG and the agent loop), scored by LLM-as-judge |
| `translation` | 4 | Sarvam-Translate across en-IN, hi-IN, ta-IN, mr-IN, bn-IN pairs, scored by judge |
| `lang_detect` | 3 | Language detection in Hindi, Tamil, Bengali scripts, scored by exact BCP-47 match |
| `round_trip` | 3 | TTS then STT on the same phrase, scored by string similarity (PASS at >= 0.80) |

### Results (run 2026-06-11)

| Category | Passed | Total | Accuracy |
|---|---|---|---|
| qa | 3 | 4 | 75.0% |
| translation | 4 | 4 | 100.0% |
| lang_detect | 3 | 3 | 100.0% |
| round_trip | 3 | 3 | 100.0% |
| **Overall** | **13** | **14** | **92.9%** |

The one qa miss was a judge disagreement on a question whose retrieved passage contained the correct answer (Devanagari for Marathi); the underlying retrieval and reasoning were correct.

<br>

## Tests

The test suite (`pytest -q`) covers retrieval chunking and cosine ordering, cache freshness/invalidation, every MCP tool with mocked sarvam_client, the scratch agent loop (happy path, malformed JSON recovery, unknown tool error, max_steps cap, multi-turn history), and the eval judge (valid JSON, malformed-then-valid re-ask, double failure, API errors). All 42 tests run without network access or model downloads in about 3 seconds.

```bash
pytest -q
```

<br>

## The two agents — what's different

### `scratch_agent.py` — framework-free

The entire mechanism is visible. The model replies with JSON; we parse it, dispatch to a tool, append the observation, and repeat. This is the loop that LangGraph runs for you — building it once by hand is how you understand what a framework actually does.

```json
{"tool": "search_knowledge", "args": {"query": "Marathi script"}}
{"tool": "answer_question", "args": {"question": "..."}}
{"final": "Marathi is written in Devanagari.", "audio_path": "reply.wav"}
```

Handles: malformed JSON (re-prompts with the contract), unknown tools (reports available tools), `max_steps` cap.

### `graph_agent.py` — LangGraph

The same behaviour, but LangGraph manages the state machine, the tool-call loop, and retries. `sarvam-30b` via its OpenAI-compatible endpoint supports native tool calling — no hand-written JSON protocol needed. MemorySaver provides conversation memory keyed by `thread_id`.

Both agents connect to the **same `mcp_server.py`** over stdio.

<br>

## Design decisions

**Why MCP instead of calling the functions directly?**
Clean, reusable boundary. The same server backs the scratch loop, the LangGraph agent, and anything else — tested in isolation with the inspector before any agent touches it.

**Why two agents?**
To make the contrast explicit. The scratch loop shows the mechanism; LangGraph shows what the framework automates. Building it by hand earns the right to say you can author agents without a framework.

**Why a single `sarvam_client.py`?**
All Sarvam-specific request shapes, model IDs, and response fields live in one file. If Sarvam changes a field name, exactly one file changes.

**Why local embeddings for RAG?**
No additional API key, no cost per query, no latency beyond the first index build. The multilingual MiniLM model handles Hindi, Tamil, and Marathi queries against English documents without translation.

**Why Streamlit for the web interface?**
Minimal code on top of the existing agent. `st.audio_input` gives a real mic widget with no JavaScript, `st.cache_resource` keeps the embedding model loaded across requests, and Render auto-deploys from GitHub on every push.

<br>

## Failure modes handled

| Failure | Handling |
|---|---|
| Malformed JSON from the model | Re-prompt with the JSON contract; retry up to `max_steps` |
| Unknown tool name in model output | Return available tool names as the observation |
| No relevant knowledge found | `search_knowledge` returns `NO_RELEVANT_KNOWLEDGE_FOUND`; agent falls back to `answer_question` |
| Wrong language detection | STT-detected language is preferred; `translate(auto)` as fallback |
| API errors | Surfaced as tool-call errors; step cap prevents runaway loops |

<br>

## Language codes supported

`hi-IN` Hindi · `mr-IN` Marathi · `ta-IN` Tamil · `te-IN` Telugu · `bn-IN` Bengali · `gu-IN` Gujarati · `kn-IN` Kannada · `ml-IN` Malayalam · `pa-IN` Punjabi · `od-IN` Odia · `en-IN` English (Indian)

<br>

## Limitations

- **Streaming** — TTS and STT are request/response, not streamed. Both Saaras and Bulbul support WebSocket streaming for lower latency; not wired up here.
- **Observability** — tool calls print to stdout but are not traced to any structured logging system. LangSmith or a simple spans table would make debugging easier at scale.

<br>

---

Built with [Sarvam AI](https://sarvam.ai) APIs · [dashboard.sarvam.ai](https://dashboard.sarvam.ai) for your free key
