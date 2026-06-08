"""
eval/run_eval.py — evaluate Setu's core capabilities against a labeled dataset.

Categories:
  answer      — question answered by the chat model, scored by LLM-as-judge
  translation — text translated, scored by LLM-as-judge
  lang_detect — language detection, scored by exact match
  round_trip  — synthesize then transcribe, scored by string similarity

Run from the repo root:
  python eval/run_eval.py
"""

import json
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

# Allow imports from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import sarvam_client as sc

DATASET   = Path(__file__).parent / "dataset.json"
RESULTS   = Path(__file__).parent / "results.json"
THRESHOLD = 0.6   # minimum similarity ratio for round-trip pass


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def _judge(reference: str, candidate: str) -> tuple[bool, str]:
    """Ask the Sarvam chat model to PASS or FAIL a candidate answer."""
    prompt = (
        "You are a strict evaluator. Compare the reference with the candidate.\n"
        "Reply with exactly: PASS: <one-line reason>  or  FAIL: <one-line reason>\n\n"
        f"Reference: {reference}\n"
        f"Candidate: {candidate}"
    )
    try:
        reply = sc.chat([{"role": "user", "content": prompt}]).strip()
    except Exception as exc:
        return False, f"judge API error: {exc}"
    upper = reply.upper()
    passed = upper.startswith("PASS")
    reason = reply.split(":", 1)[1].strip() if ":" in reply else reply
    return passed, reason


# ---------------------------------------------------------------------------
# Per-category runners
# ---------------------------------------------------------------------------

def _run_answer(case: dict) -> dict:
    try:
        candidate = sc.chat([{"role": "user", "content": case["question"]}])
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    passed, reason = _judge(case["reference"], candidate)
    return {"candidate": candidate, "passed": passed, "reason": reason}


def _run_translation(case: dict) -> dict:
    try:
        candidate = sc.translate(
            case["text"],
            target_language_code=case["target_lang"],
            source_language_code=case["source_lang"],
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    passed, reason = _judge(case["reference"], candidate)
    return {"candidate": candidate, "passed": passed, "reason": reason}


def _run_lang_detect(case: dict) -> dict:
    prompt = (
        "Identify the language of the following text and reply with ONLY its BCP-47 code "
        "(e.g. hi-IN, ta-IN, mr-IN, en-IN). No explanation.\n\n" + case["text"]
    )
    try:
        detected = sc.chat([{"role": "user", "content": prompt}]).strip()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    passed = detected.lower() == case["expected_lang"].lower()
    return {"detected": detected, "expected": case["expected_lang"], "passed": passed}


def _run_round_trip(case: dict) -> dict:
    wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sc.synthesize(case["text"], target_language_code=case["lang"], out_path=str(wav))
        tr = sc.transcribe(str(wav))
        transcribed = tr.text
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        if wav.exists():
            wav.unlink()
    ratio = SequenceMatcher(None, case["text"].strip(), transcribed.strip()).ratio()
    passed = ratio >= THRESHOLD
    return {"original": case["text"], "transcribed": transcribed, "similarity": round(ratio, 3), "passed": passed}


RUNNERS = {
    "answer":      _run_answer,
    "translation": _run_translation,
    "lang_detect": _run_lang_detect,
    "round_trip":  _run_round_trip,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    records   = []
    by_cat: dict[str, dict] = {}

    print(f"Running {len(cases)} eval cases...\n")

    for i, case in enumerate(cases, 1):
        cat    = case["type"]
        runner = RUNNERS.get(cat)
        if runner is None:
            print(f"  [{i:02d}] SKIP — unknown type '{cat}'")
            continue

        label = case.get("question") or case.get("text") or ""
        print(f"  [{i:02d}] {cat:12s}  {label[:55]}")

        outcome = runner(case)
        passed  = outcome.get("passed", False)
        status  = "PASS" if passed else ("ERROR" if "error" in outcome else "FAIL")
        print(f"         {status}  {outcome.get('reason') or outcome.get('detected') or outcome.get('similarity', '')}")

        record = {"type": cat, "input": case, "result": outcome}
        records.append(record)

        if cat not in by_cat:
            by_cat[cat] = {"passed": 0, "total": 0}
        by_cat[cat]["total"]  += 1
        if passed:
            by_cat[cat]["passed"] += 1

    # Summary
    total_passed = sum(v["passed"] for v in by_cat.values())
    total_cases  = sum(v["total"]  for v in by_cat.values())
    overall      = total_passed / total_cases if total_cases else 0.0

    print("\n" + "=" * 50)
    print(f"  {'Category':<14} {'Passed':>6}  {'Total':>5}  {'Accuracy':>8}")
    print("  " + "-" * 38)
    for cat, counts in by_cat.items():
        acc = counts["passed"] / counts["total"] if counts["total"] else 0
        print(f"  {cat:<14} {counts['passed']:>6}  {counts['total']:>5}  {acc:>7.1%}")
    print("  " + "-" * 38)
    print(f"  {'Overall':<14} {total_passed:>6}  {total_cases:>5}  {overall:>7.1%}")
    print("=" * 50)

    results = {
        "overall": round(overall, 4),
        "by_category": {
            cat: {**v, "accuracy": round(v["passed"] / v["total"], 4) if v["total"] else 0}
            for cat, v in by_cat.items()
        },
        "cases": records,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
