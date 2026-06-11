"""
tests/test_tools.py — unit tests for each MCP tool in mcp_server.py.

All Sarvam API calls are mocked so no network access is needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, patch as mock_patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp_server
import sarvam_client as sc


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

def test_answer_question_returns_chat_result():
    with patch.object(sc, "chat", return_value="New Delhi is the capital of India."):
        result = mcp_server.answer_question("What is the capital of India?")
    assert result == "New Delhi is the capital of India."


def test_answer_question_passes_message_to_chat():
    captured = {}
    def fake_chat(messages, **kw):
        captured["messages"] = messages
        return "answer"
    with patch.object(sc, "chat", side_effect=fake_chat):
        mcp_server.answer_question("test question")
    assert captured["messages"][0]["content"] == "test question"
    assert captured["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# translate_text
# ---------------------------------------------------------------------------

def test_translate_text_returns_translation():
    with patch.object(sc, "translate", return_value="नमस्ते"):
        result = mcp_server.translate_text("Hello", "en-IN", "hi-IN")
    assert result == "नमस्ते"


def test_translate_text_passes_correct_args():
    captured = {}
    def fake_translate(text, target_language_code, source_language_code):
        captured.update(locals())
        return "translated"
    with patch.object(sc, "translate", side_effect=fake_translate):
        mcp_server.translate_text("Hello", "auto", "hi-IN")
    assert captured["text"] == "Hello"
    assert captured["source_language_code"] == "auto"
    assert captured["target_language_code"] == "hi-IN"


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

def test_detect_language_returns_stripped_code():
    with patch.object(sc, "chat", return_value="  hi-IN  "):
        result = mcp_server.detect_language("नमस्ते")
    assert result == "hi-IN"


def test_detect_language_sends_text_in_prompt():
    captured = {}
    def fake_chat(messages, **kw):
        captured["content"] = messages[0]["content"]
        return "ta-IN"
    with patch.object(sc, "chat", side_effect=fake_chat):
        mcp_server.detect_language("வணக்கம்")
    assert "வணக்கம்" in captured["content"]


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------

def test_transcribe_audio_formats_result():
    mock_result = MagicMock()
    mock_result.text = "नमस्ते"
    mock_result.language_code = "hi-IN"
    with patch.object(sc, "transcribe", return_value=mock_result):
        result = mcp_server.transcribe_audio("audio.wav")
    assert "नमस्ते" in result
    assert "hi-IN" in result


def test_transcribe_audio_passes_path():
    captured = {}
    mock_result = MagicMock(text="hello", language_code="en-IN")
    def fake_transcribe(path, **kw):
        captured["path"] = path
        return mock_result
    with patch.object(sc, "transcribe", side_effect=fake_transcribe):
        mcp_server.transcribe_audio("samples/test.wav")
    assert captured["path"] == "samples/test.wav"


# ---------------------------------------------------------------------------
# synthesize_speech
# ---------------------------------------------------------------------------

def test_synthesize_speech_returns_path():
    with patch.object(sc, "synthesize", return_value="reply.wav"):
        result = mcp_server.synthesize_speech("नमस्ते", "hi-IN")
    assert result == "reply.wav"


def test_synthesize_speech_passes_correct_args():
    captured = {}
    def fake_synthesize(text, target_language_code, **kw):
        captured.update({"text": text, "lang": target_language_code})
        return "out.wav"
    with patch.object(sc, "synthesize", side_effect=fake_synthesize):
        mcp_server.synthesize_speech("hello", "en-IN")
    assert captured["text"] == "hello"
    assert captured["lang"] == "en-IN"


