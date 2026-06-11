"""
tests/test_scratch_agent.py — unit tests for the scratch agent loop.

sc.chat is mocked throughout so no network access is needed.
Tool functions that call sc.chat directly are also covered by the mock.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scratch_agent
import sarvam_client as sc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _responses(*replies):
    """Return a side_effect iterator over a fixed list of replies."""
    it = iter(replies)
    return lambda messages, **kw: next(it)


# ---------------------------------------------------------------------------
# Happy path: immediate final answer (no tool call)
# ---------------------------------------------------------------------------

def test_immediate_final_answer():
    with patch.object(sc, "chat", side_effect=_responses('{"final": "42"}')):
        result, history = scratch_agent.run("What is 6 times 7?")
    assert result["final"] == "42"
    assert result["audio_path"] is None


def test_history_starts_with_system_message():
    with patch.object(sc, "chat", side_effect=_responses('{"final": "done"}')):
        _, history = scratch_agent.run("hi")
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "hi"


# ---------------------------------------------------------------------------
# Tool call then final answer
# ---------------------------------------------------------------------------

def test_tool_call_then_final():
    # call 1: agent proposes answer_question
    # call 2: _answer_question calls sc.chat internally
    # call 3: agent emits final
    responses = _responses(
        '{"tool": "answer_question", "args": {"question": "capital of India?"}}',
        "New Delhi",
        '{"final": "New Delhi is the capital.", "audio_path": null}',
    )
    with patch.object(sc, "chat", side_effect=responses):
        result, history = scratch_agent.run("What is the capital of India?")
    assert "New Delhi" in result["final"]


def test_tool_call_result_appended_to_history():
    responses = _responses(
        '{"tool": "answer_question", "args": {"question": "capital?"}}',
        "New Delhi",
        '{"final": "New Delhi."}',
    )
    with patch.object(sc, "chat", side_effect=responses):
        _, history = scratch_agent.run("capital?")
    # History should contain an Observation message
    contents = [m["content"] for m in history]
    assert any("Observation" in c for c in contents)


# ---------------------------------------------------------------------------
# Malformed JSON recovery
# ---------------------------------------------------------------------------

def test_malformed_json_reprompts_then_recovers():
    responses = _responses(
        "this is not json at all",
        '{"final": "recovered answer"}',
    )
    with patch.object(sc, "chat", side_effect=responses):
        result, history = scratch_agent.run("test query")
    assert result["final"] == "recovered answer"


def test_malformed_json_adds_correction_message():
    responses = _responses(
        "not json",
        '{"final": "ok"}',
    )
    with patch.object(sc, "chat", side_effect=responses):
        _, history = scratch_agent.run("test")
    # A user message prompting for valid JSON should be in history
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    assert any("valid JSON" in m or "not valid JSON" in m for m in user_msgs)


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_reports_available_tools():
    responses = _responses(
        '{"tool": "nonexistent_tool", "args": {}}',
        '{"final": "done"}',
    )
    with patch.object(sc, "chat", side_effect=responses):
        _, history = scratch_agent.run("test")
    obs_msgs = [m["content"] for m in history if "Observation" in m.get("content", "")]
    assert any("Unknown tool" in m or "nonexistent_tool" in m for m in obs_msgs)


# ---------------------------------------------------------------------------
# Max steps cap
# ---------------------------------------------------------------------------

def test_max_steps_cap_stops_loop():
    # Always return a tool call — loop should hit max_steps and stop
    always_tool = lambda msgs, **kw: '{"tool": "answer_question", "args": {"question": "q"}}'

    def side_effect(msgs, **kw):
        # Even tool calls inside _answer_question get this, so return a string
        return '{"tool": "answer_question", "args": {"question": "q"}}'

    with patch.object(sc, "chat", side_effect=side_effect):
        result, _ = scratch_agent.run("loop forever", max_steps=3)
    assert "max steps" in result["final"].lower()


# ---------------------------------------------------------------------------
# History continuity across turns
# ---------------------------------------------------------------------------

def test_history_carries_over_across_turns():
    with patch.object(sc, "chat", side_effect=_responses('{"final": "Paris"}')):
        result1, history = scratch_agent.run("What is the capital of France?")

    # Second turn: pass history back — history should include prior exchange
    with patch.object(sc, "chat", side_effect=_responses('{"final": "It has the Eiffel Tower"}')):
        result2, history2 = scratch_agent.run("What is it famous for?", history=history)

    # The second turn's messages should include the first turn's content
    all_contents = " ".join(m["content"] for m in history2)
    assert "capital of France" in all_contents
    assert "What is it famous for?" in all_contents


def test_run_without_history_uses_system_prompt():
    with patch.object(sc, "chat", side_effect=_responses('{"final": "hi"}')):
        _, history = scratch_agent.run("hello", history=None)
    assert history[0]["role"] == "system"
    assert "Setu" in history[0]["content"]


def test_run_with_existing_history_does_not_duplicate_system():
    # Pre-built history already has system message
    prior_history = [
        {"role": "system", "content": scratch_agent.SYSTEM},
        {"role": "user",   "content": "first question"},
        {"role": "assistant", "content": '{"final": "first answer"}'},
    ]
    with patch.object(sc, "chat", side_effect=_responses('{"final": "second answer"}')):
        _, history = scratch_agent.run("second question", history=prior_history)
    system_msgs = [m for m in history if m["role"] == "system"]
    assert len(system_msgs) == 1  # no duplicate system message
