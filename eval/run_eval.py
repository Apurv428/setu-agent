"""
eval/run_eval.py — evaluate Setu's core capabilities against a labeled dataset.

Categories:
  qa          — question answered via the scratch agent (exercises RAG path),
                scored by LLM-as-judge
  translation — text translated by sarvam_client.translate, scored by judge
  lang_detect — language detection, scored by exact BCP-47 match
  round_trip  — synthesize then transcribe, scored by string similarity

Run from the repo root:
  python eval/run_eval.py
  python eval/run_eval.py --category qa
  python eval/run_eval.py --verbose
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sarvam_client as sc

DATASET = Path(__file__).parent / "dataset.json"
RESULTS = Path(__file__).parent / "results.json"
ROUND_TRIP_PASS_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# LLM-as-judge (strict JSON output)
# ---------------------------------------------------------------------------

def _judge(reference: str, candidate: str) -> tuple[bool, str]:
    """Ask the Sarvam chat model to PASS or FAIL a candidate answer.

    Demands strict JSON: {"verdict": "PASS" or "FAIL", "reason": "<one line>"}
    On malformed output, re-asks once. On second failure, counts as FAIL.
    """
    prompt = (
        "You are a strict evaluator. Does the candidate answer convey the same "
        "meaning as the reference? Reply with ONLY this JSON and nothing else:\n"
        '{"verdict": "PASS", "reason": "<one line>"}\n'
        "or\n"
        '{"verdict": "FAIL", "reason": "<one line>"}\n\n'
        f"Reference: {reference}\n"
        f"Candidate: {candidate}"
    )

    def _parse_judge(reply: str) -> tuple[bool, str] | None:
        text = reply.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        try:
            obj = json.loads(text)
            verdict = str(obj.get("verdict", "")).strip().upper()
            reason = str(obj.get("reason", "")).strip()
            if verdict in ("PASS", "FAIL"):
                return verdict == "PASS", reason
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    try:
        reply = sc.chat([{"role": "user", "content": prompt}])
    except Exception as exc:
        return False, f"judge_api_error: {exc}"

    parsed = _parse_judge(reply)
    if parsed is not None:
        return parsed

    # Re-ask once
    try:
        retry_prompt = (
            "Your previous reply was not valid JSON. Reply with ONLY:\n"
            '{"verdict": "PASS", "reason": "..."} or {"verdict": "FAIL", "reason": "..."}\n\n'
            f"Reference: {reference}\nCandidate: {candidate}"
        )
        reply2 = sc.chat([{"role": "user", "content": retry_prompt}])
    except Exception as exc:
        return False, f"judge_api_error: {exc}"

    parsed2 = _parse_judge(reply2)
    if parsed2 is not None:
        return parsed2

    return False, "judge_parse_error"


# ---------------------------------------------------------------------------
# Per-category runners
# ---------------------------------------------------------------------------

def _run_qa(case: dict) -> dict:
    """Run qa cases through the scratch agent so RAG is exercised."""
    try:
        from scratch_agent import run as agent_run
        result, _ = agent_run(case["question"])
        candidate = result.get("final", "")
    except Exception as exc:
        return {"passed": False, "error": str(exc)}
    passed, reason = _judge(case["reference_answer"], candidate)
    return {"candidate": candidate, "passed": passed, "reason": reason}


def _run_translation(case: dict) -> dict:
    try:
        candidate = sc.translate(
            case["text"],
            target_language_code=case["target_lang"],
            source_language_code=case["source_lang"],
        )
    except Exception as exc:
        return {"passed": False, "error": str(exc)}
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
        return {"passed": False, "error": str(exc)}
    passed = detected.lower() == case["expected_lang"].lower()
    return {"detected": detected, "expected": case["expected_lang"], "passed": passed}


def _run_round_trip(case: dict) -> dict:
    wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sc.synthesize(case["text"], target_language_code=case["lang"], out_path=str(wav))
        tr = sc.transcribe(str(wav))
        transcribed = tr.text
    except Exception as exc:
        return {"passed": False, "error": str(exc)}
    finally:
        if wav.exists():
            wav.unlink()
    ratio = SequenceMatcher(None, case["text"].strip(), transcribed.strip()).ratio()
    passed = ratio >= ROUND_TRIP_PASS_THRESHOLD
    return {
        "original": case["text"],
        "transcribed": transcribed,
        "similarity": round(ratio, 3),
        "passed": passed,
    }


RUNNERS = {
    "qa":          _run_qa,
    "translation": _run_translation,
    "lang_detect": _run_lang_detect,
    "round_trip":  _run_round_trip,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Setu eval suite")
    parser.add_argument("--category", help="Run only this category (qa/translation/lang_detect/round_trip)")
    parser.add_argument("--verbose", action="store_true", help="Print each case result in detail")
    args = parser.parse_args()

    all_cases = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if args.category is None or c["category"] == args.category]

    if not cases:
        print(f"No cases found for category '{args.category}'.")
        sys.exit(1)

    print(f"Running {len(cases)} eval cases...\n")

    records: list[dict] = []
    by_cat: dict[str, dict] = {}

    for case in cases:
        cat = case["category"]
        runner = RUNNERS.get(cat)
        if runner is None:
            print(f"  [{case['id']}] SKIP — unknown category '{cat}'")
            continue

        label = case.get("question") or case.get("text") or case["id"]
        safe_label = label[:55].encode("ascii", errors="replace").decode("ascii")
        print(f"  [{case['id']}] {cat:<12}  {safe_label}", end="", flush=True)

        outcome = runner(case)
        passed = outcome.get("passed", False)

        if "error" in outcome:
            status = "ERROR"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  {status}")

        if args.verbose:
            def _s(v: str) -> str:
                return str(v).encode("ascii", errors="replace").decode("ascii")
            if "candidate" in outcome:
                print(f"           candidate  : {_s(outcome['candidate'])[:120]}")
            if "reason" in outcome:
                print(f"           reason     : {_s(outcome['reason'])}")
            if "detected" in outcome:
                print(f"           detected   : {_s(outcome['detected'])}  (expected {_s(outcome['expected'])})")
            if "similarity" in outcome:
                print(f"           similarity : {outcome['similarity']}  (original: {_s(outcome.get('original',''))} | transcribed: {_s(outcome.get('transcribed',''))})")
            if "error" in outcome:
                print(f"           error      : {_s(outcome['error'])}")

        record = {"id": case["id"], "category": cat, "input": case, "result": outcome}
        records.append(record)

        if cat not in by_cat:
            by_cat[cat] = {"passed": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if passed:
            by_cat[cat]["passed"] += 1

    total_passed = sum(v["passed"] for v in by_cat.values())
    total_cases = sum(v["total"] for v in by_cat.values())
    overall = total_passed / total_cases if total_cases else 0.0

    print("\n" + "=" * 52)
    print(f"  {'Category':<14}  {'Passed':>6}  {'Total':>5}  {'Accuracy':>8}")
    print("  " + "-" * 40)
    for cat, counts in by_cat.items():
        acc = counts["passed"] / counts["total"] if counts["total"] else 0
        print(f"  {cat:<14}  {counts['passed']:>6}  {counts['total']:>5}  {acc:>7.1%}")
    print("  " + "-" * 40)
    print(f"  {'Overall':<14}  {total_passed:>6}  {total_cases:>5}  {overall:>7.1%}")
    print("=" * 52)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
