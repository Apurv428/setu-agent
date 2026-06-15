"""
streamlit_app.py — web interface for Setu voice agent.

Run:
  streamlit run streamlit_app.py

Supports text input and mic recording. Conversation memory is kept
across turns within the browser session using st.session_state.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Setu — Multilingual Voice Agent",
    page_icon="🌉",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Warm the retrieval index once per server process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge index...")
def _load_retrieval():
    import retrieval
    retrieval._ensure_index()
    return True

_load_retrieval()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = None          # scratch_agent message list
if "chat" not in st.session_state:
    st.session_state.chat = []               # [{role, content, steps, audio}]
if "last_audio_key" not in st.session_state:
    st.session_state.last_audio_key = None   # dedup audio submissions

# ---------------------------------------------------------------------------
# Agent runner (captures tool-step trace from stdout)
# ---------------------------------------------------------------------------

def _run_agent(user_input: str) -> tuple[str, str | None, str]:
    """
    Run the scratch agent and return (final_answer, audio_path, step_trace).
    step_trace is the stdout printed during the run (tool call lines).
    """
    from scratch_agent import run

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result, new_history = run(user_input, history=st.session_state.history)

    st.session_state.history = new_history
    return result["final"], result.get("audio_path"), buf.getvalue().strip()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("Tool calls", expanded=False):
                    st.code(msg["steps"], language="text")
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")


def _submit(user_text: str, audio_bytes: bytes | None = None):
    """Process one turn: show user message, run agent, show reply."""
    if audio_bytes:
        # Save audio bytes (already WAV from st.audio_input) to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            wav_path = f.name
        display_text = "🎤 *Voice message*"
        agent_input = wav_path
    else:
        display_text = user_text
        agent_input = user_text

    st.session_state.chat.append({"role": "user", "content": display_text})

    with st.spinner("Setu is thinking..."):
        try:
            answer, audio_path, steps = _run_agent(agent_input)
        except Exception as exc:
            st.session_state.chat.append({
                "role": "assistant",
                "content": f"Error: {exc}",
                "steps": "",
                "audio": None,
            })
            return

    # Read synthesized audio bytes so we can pass them to st.audio
    audio_bytes_out = None
    if audio_path and Path(audio_path).exists():
        audio_bytes_out = Path(audio_path).read_bytes()

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
        "audio": audio_bytes_out,
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🌉 Setu")
    st.caption("Multilingual voice agent powered by Sarvam AI")
    st.divider()

    st.markdown("**Supported languages**")
    st.markdown(
        "Hindi · Marathi · Tamil · Telugu · Bengali\n"
        "Gujarati · Kannada · Malayalam · Punjabi · Odia · English"
    )
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "1. Type a question or record your voice.\n"
        "2. The agent searches the knowledge base, then reasons with `sarvam-30b`.\n"
        "3. The answer is spoken back in your language via Bulbul v3 TTS."
    )
    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = None
        st.session_state.chat = []
        st.session_state.last_audio_key = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown("## Ask Setu")
st.caption("Speak or type in any Indian language. Setu will reply in the same language.")

_render_chat()

st.divider()

tab_text, tab_voice = st.tabs(["Text", "Voice"])

with tab_text:
    with st.form("text_form", clear_on_submit=True):
        text_input = st.text_input(
            "Your question",
            placeholder="Which script is used to write Marathi?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and text_input.strip():
        _submit(text_input.strip())
        st.rerun()

with tab_voice:
    st.caption("Click the microphone, speak, then click Stop. The agent runs automatically.")
    audio = st.audio_input("Record your question", label_visibility="collapsed")
    if audio is not None:
        audio_bytes = audio.read()
        audio_id = hash(audio_bytes)
        if audio_id != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_id
            _submit(user_text="", audio_bytes=audio_bytes)
            st.rerun()
