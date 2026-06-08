"""
scratch_agent.py — framework-free tool-calling agent loop.

No LangChain, no LangGraph, no agent libraries.
The entire mechanism is visible: the model proposes an action as JSON,
we parse it, dispatch to a local tool function, append the observation,
and repeat until the model emits a final answer or we hit max_steps.

JSON protocol (the only two valid responses from the model):
  {"tool": "<name>", "args": {<key>: <value>}}   -> call a tool
  {"final": "<answer>", "audio_path": "<path>"}  -> done (audio_path optional)

Multi-turn usage:
  result, history = run("first question")
  result, history = run("follow-up", history=history)
"""

import json
import sys
import sarvam_client as sc

# ---------------------------------------------------------------------------
# Tool implementations (same surface as mcp_server.py tools)
# ---------------------------------------------------------------------------

def _transcribe_audio(audio_path: str) -> str:
    r = sc.transcribe(audio_path)
    return f"Transcript: {r.text}\nLanguage: {r.language_code}"


def _detect_language(text: str) -> str:
    return sc.chat([{
        "role": "user",
        "content": (
            "Identify the language of the following text and reply with ONLY its BCP-47 code "
            "(e.g. hi-IN, ta-IN, mr-IN, en-IN). No explanation.\n\n" + text
        ),
    }]).strip()


def _translate_text(text: str, source_language_code: str, target_language_code: str) -> str:
    return sc.translate(text, target_language_code=target_language_code,
                        source_language_code=source_language_code)


def _answer_question(question: str) -> str:
    return sc.chat([{"role": "user", "content": question}])


def _synthesize_speech(text: str, target_language_code: str) -> str:
    return sc.synthesize(text, target_language_code=target_language_code)


def _search_knowledge(query: str) -> str:
    import retrieval
    try:
        passages = retrieval.search(query, k=3)
    except Exception as exc:
        return f"Knowledge search unavailable: {exc}"
    if not passages:
        return "No relevant passages found."
    return "\n\n---\n\n".join(passages)


TOOLS: dict = {
    "transcribe_audio":  _transcribe_audio,
    "detect_language":   _detect_language,
    "translate_text":    _translate_text,
    "answer_question":   _answer_question,
    "synthesize_speech": _synthesize_speech,
    "search_knowledge":  _search_knowledge,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = """\
You are Setu, a multilingual voice assistant powered by Sarvam AI.
You have access to these tools:

  transcribe_audio(audio_path: str)
    Transcribes a WAV file. Returns transcript text and detected language code.

  detect_language(text: str)
    Returns the BCP-47 language code of the text (e.g. hi-IN, ta-IN, en-IN).

  translate_text(text: str, source_language_code: str, target_language_code: str)
    Translates text. Pass "auto" as source_language_code to auto-detect.

  answer_question(question: str)
    Answers a question using the Sarvam chat model.

  synthesize_speech(text: str, target_language_code: str)
    Converts text to speech; returns the path to the saved WAV file.

  search_knowledge(query: str)
    Search a local knowledge base about Indian languages, Indic scripts, and
    Sarvam AI models. Call this before answer_question when the question is
    knowledge-grounded (about Indian languages, their history, scripts, or
    Sarvam's capabilities).

Instructions:
- If the input is an audio file path, start by transcribing it.
- Detect or infer the user's language and reply in that same language.
- Translate to English before calling answer_question if that helps accuracy.
- Always call synthesize_speech as the last step so the reply is spoken.
- Reply with ONLY valid JSON — no prose, no markdown fences. Use:
    {"tool": "<name>", "args": {"<key>": "<value>"}}
  or, when finished:
    {"final": "<answer text>", "audio_path": "<path returned by synthesize_speech>"}
"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run(
    user_input: str,
    history: list[dict] | None = None,
    max_steps: int = 10,
) -> tuple[dict, list[dict]]:
    """
    Run the agent loop on a text query or audio file path.

    Pass history from a previous call to maintain conversation context across turns.
    Returns (result_dict, updated_history) where result_dict has keys
    "final" (answer string) and "audio_path" (wav path or None).
    """
    if history is None:
        messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    else:
        messages = list(history)  # copy so caller's list is not mutated

    messages.append({"role": "user", "content": user_input})

    for step in range(max_steps):
        raw = sc.chat(messages)
        messages.append({"role": "assistant", "content": raw})

        # Strip markdown code fences the model sometimes adds
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

        try:
            action = json.loads(text)
        except json.JSONDecodeError:
            messages.append({
                "role": "user",
                "content": (
                    "Your last reply was not valid JSON. "
                    'Reply with ONLY: {"tool": "...", "args": {...}} or {"final": "..."}'
                ),
            })
            continue

        if "final" in action:
            result = {
                "final": action["final"],
                "audio_path": action.get("audio_path"),
            }
            return result, messages

        tool_name = action.get("tool")
        if not tool_name:
            messages.append({
                "role": "user",
                "content": 'Missing "tool" key. Use {"tool": "...", "args": {...}}.',
            })
            continue

        args = action.get("args", {})
        if tool_name not in TOOLS:
            observation = f"Unknown tool '{tool_name}'. Available: {list(TOOLS)}"
        else:
            try:
                observation = str(TOOLS[tool_name](**args))
            except Exception as exc:
                observation = f"Error in {tool_name}: {exc}"

        print(f"  step {step + 1}: {tool_name}({args})")
        print(f"           -> {str(observation)[:200]}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    result = {"final": "Stopped: max steps reached.", "audio_path": None}
    return result, messages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the capital of India? Answer in Hindi and speak the reply."
    print(f"Query: {query}\n")
    result, _ = run(query)
    print(f"\nFinal: {result['final']}")
    if result.get("audio_path"):
        print(f"Audio: {result['audio_path']}")
