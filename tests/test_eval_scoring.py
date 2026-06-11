"""
tests/test_eval_scoring.py — unit tests for eval/run_eval.py scoring logic.

No network calls. sc.chat is mocked throughout.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import sarvam_client as sc


# Import the scoring helpers from the eval module
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from run_eval import _judge, _run_round_trip


# ---------------------------------------------------------------------------
# round_trip similarity scoring
# ---------------------------------------------------------------------------

def test_round_trip_pass_high_similarity():
    """High similarity (>= 0.80) should pass."""
    with patch.object(sc, "synthesize", return_value="fake.wav"), \
         patch.object(sc, "transcribe") as mock_tr, \
         patch("pathlib.Path.exists", return_value=False):
        mock_result = type("R", (), {"text": "namaste", "language_code": "hi-IN"})()
        mock_tr.return_value = mock_result
        case = {"text": "namaste", "lang": "hi-IN"}
        result = _run_round_trip(case)
    assert result["passed"] is True
    assert result["similarity"] >= 0.80


def test_round_trip_fail_low_similarity():
    """Low similarity (< 0.80) should fail."""
    with patch.object(sc, "synthesize", return_value="fake.wav"), \
         patch.object(sc, "transcribe") as mock_tr, \
         patch("pathlib.Path.exists", return_value=False):
        mock_result = type("R", (), {"text": "completely different text here xyz", "language_code": "hi-IN"})()
        mock_tr.return_value = mock_result
        case = {"text": "namaste", "lang": "hi-IN"}
        result = _run_round_trip(case)
    assert result["passed"] is False


def test_round_trip_api_error_counts_as_fail():
    """API errors should return passed=False with error recorded."""
    with patch.object(sc, "synthesize", side_effect=Exception("network error")):
        case = {"text": "test", "lang": "hi-IN"}
        result = _run_round_trip(case)
    assert result["passed"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Judge JSON parsing — valid input
# ---------------------------------------------------------------------------

def test_judge_pass_on_valid_json():
    reply = '{"verdict": "PASS", "reason": "Answers match"}'
    with patch.object(sc, "chat", return_value=reply):
        passed, reason = _judge("New Delhi", "The capital is New Delhi.")
    assert passed is True
    assert reason == "Answers match"


def test_judge_fail_on_valid_json():
    reply = '{"verdict": "FAIL", "reason": "Incorrect city named"}'
    with patch.object(sc, "chat", return_value=reply):
        passed, reason = _judge("New Delhi", "Mumbai is the capital.")
    assert passed is False
    assert reason == "Incorrect city named"


# ---------------------------------------------------------------------------
# Judge JSON parsing — malformed then valid (re-ask recovers)
# ---------------------------------------------------------------------------

def test_judge_malformed_then_valid_recovers():
    """First reply is bad JSON; second is valid → should count as PASS."""
    replies = iter([
        "PASS: looks correct",   # malformed (not JSON)
        '{"verdict": "PASS", "reason": "recovered ok"}',
    ])
    with patch.object(sc, "chat", side_effect=lambda msgs, **kw: next(replies)):
        passed, reason = _judge("ref", "candidate")
    assert passed is True
    assert reason == "recovered ok"


def test_judge_malformed_twice_counts_as_fail():
    """Both replies are bad JSON → should count as FAIL with judge_parse_error."""
    replies = iter([
        "not json at all",
        "still not json",
    ])
    with patch.object(sc, "chat", side_effect=lambda msgs, **kw: next(replies)):
        passed, reason = _judge("ref", "candidate")
    assert passed is False
    assert "judge_parse_error" in reason


# ---------------------------------------------------------------------------
# Judge — API error
# ---------------------------------------------------------------------------

def test_judge_api_error_counts_as_fail():
    with patch.object(sc, "chat", side_effect=Exception("timeout")):
        passed, reason = _judge("ref", "candidate")
    assert passed is False
    assert "judge_api_error" in reason


# ---------------------------------------------------------------------------
# Judge — markdown-fenced JSON is accepted
# ---------------------------------------------------------------------------

def test_judge_accepts_markdown_fenced_json():
    reply = "```json\n{\"verdict\": \"PASS\", \"reason\": \"correct\"}\n```"
    with patch.object(sc, "chat", return_value=reply):
        passed, reason = _judge("ref", "ref")
    assert passed is True
