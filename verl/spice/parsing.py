from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from mathruler.grader import extract_boxed_content


_REQUIRED_CHALLENGER_KEYS = {
    "analysis",
    "question",
    "intermediate_results",
    "answer",
    "solving_time_estimate",
    "required_concepts",
    "potential_errors",
}

# Enforce: float answers must be written with a decimal point in the JSON source.
_ANSWER_FLOAT_LITERAL_RE = re.compile(
    r"\"answer\"\s*:\s*([-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?)\b"
)

_CATEGORICAL_OPTION_RE = re.compile(r"(?m)^\s*\(?([A-D])\)?[).:]\s+")


@dataclass(frozen=True)
class ParsedQA:
    question: str
    answer: str
    valid: bool
    reason: Optional[str] = None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    if not _is_number(x):
        return False
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def parse_challenger_output(
    text: str,
    *,
    expected_answer_type: str = "integer",
    max_question_chars: int = 2000,
) -> ParsedQA:
    if not isinstance(text, str):
        return ParsedQA(question="", answer="", valid=False, reason="non-string")

    raw = text.strip()
    if not raw:
        return ParsedQA(question="", answer="", valid=False, reason="empty")
    if not (raw.startswith("{") and raw.endswith("}")):
        return ParsedQA(question="", answer="", valid=False, reason="not_json_object")

    # STRICT: must be exactly one JSON object matching the schema in `verl/spice/prompting.py`.
    try:
        obj = json.loads(raw)
    except Exception:
        return ParsedQA(question="", answer="", valid=False, reason="invalid_json")

    if not isinstance(obj, dict):
        return ParsedQA(question="", answer="", valid=False, reason="json_not_object")

    keys = set(obj.keys())
    if keys != _REQUIRED_CHALLENGER_KEYS:
        missing = sorted(_REQUIRED_CHALLENGER_KEYS - keys)
        extra = sorted(keys - _REQUIRED_CHALLENGER_KEYS)
        if missing and extra:
            reason = f"keys_mismatch(missing={missing},extra={extra})"
        elif missing:
            reason = f"missing_keys({missing})"
        else:
            reason = f"extra_keys({extra})"
        return ParsedQA(question="", answer="", valid=False, reason=reason)

    analysis = obj.get("analysis")
    question = obj.get("question")
    intermediate_results = obj.get("intermediate_results")
    answer = obj.get("answer")
    solving_time_estimate = obj.get("solving_time_estimate")
    required_concepts = obj.get("required_concepts")
    potential_errors = obj.get("potential_errors")

    if not isinstance(analysis, str):
        return ParsedQA(question="", answer="", valid=False, reason="analysis_not_string")
    if not isinstance(question, str) or not question.strip():
        return ParsedQA(question="", answer="", valid=False, reason="question_not_string")
    question = question.strip()
    if max_question_chars and len(question) > max_question_chars:
        question = question[:max_question_chars].rstrip()

    if not isinstance(intermediate_results, dict) or not intermediate_results:
        return ParsedQA(question=question, answer="", valid=False, reason="intermediate_results_not_object")
    for k, v in intermediate_results.items():
        if not isinstance(k, str) or not k.strip():
            return ParsedQA(question=question, answer="", valid=False, reason="intermediate_results_bad_key")
        if not isinstance(v, str) or not v.strip():
            return ParsedQA(question=question, answer="", valid=False, reason="intermediate_results_bad_value")

    if not _is_finite_number(solving_time_estimate):
        return ParsedQA(question=question, answer="", valid=False, reason="solving_time_estimate_not_number")

    if not isinstance(required_concepts, list) or not (1 <= len(required_concepts) <= 10):
        return ParsedQA(question=question, answer="", valid=False, reason="required_concepts_not_array")
    if not all(isinstance(x, str) and x.strip() for x in required_concepts):
        return ParsedQA(question=question, answer="", valid=False, reason="required_concepts_bad_item")

    if not isinstance(potential_errors, list) or not (1 <= len(potential_errors) <= 10):
        return ParsedQA(question=question, answer="", valid=False, reason="potential_errors_not_array")
    if not all(isinstance(x, str) and x.strip() for x in potential_errors):
        return ParsedQA(question=question, answer="", valid=False, reason="potential_errors_bad_item")

    at = (expected_answer_type or "integer").strip().lower()
    if at not in {"integer", "float", "categorical"}:
        at = "integer"

    if at == "integer":
        if not isinstance(answer, int) or isinstance(answer, bool):
            return ParsedQA(question=question, answer="", valid=False, reason="answer_not_integer")
        if not _is_finite_number(answer):
            return ParsedQA(question=question, answer="", valid=False, reason="answer_not_number")
        return ParsedQA(question=question, answer=str(answer), valid=True, reason=None)
    elif at == "float":
        if not isinstance(answer, float) or isinstance(answer, bool):
            return ParsedQA(question=question, answer="", valid=False, reason="answer_not_float")
        # Must be written with a decimal point in the JSON source (not exponent-only).
        if _ANSWER_FLOAT_LITERAL_RE.search(raw) is None:
            return ParsedQA(question=question, answer="", valid=False, reason="answer_float_missing_decimal_point")
        if not _is_finite_number(answer):
            return ParsedQA(question=question, answer="", valid=False, reason="answer_not_number")
        return ParsedQA(question=question, answer=str(answer), valid=True, reason=None)
    else:
        # categorical: "answer" must be a single uppercase letter, matching one of the options.
        if not isinstance(answer, str) or not answer.strip():
            return ParsedQA(question=question, answer="", valid=False, reason="answer_not_categorical")
        ans = answer.strip().upper()
        if not re.fullmatch(r"[A-D]", ans):
            return ParsedQA(question=question, answer="", valid=False, reason="answer_categorical_not_a_d")
        opts = {m.group(1).upper() for m in _CATEGORICAL_OPTION_RE.finditer(question)}
        if opts != {"A", "B", "C", "D"}:
            return ParsedQA(question=question, answer="", valid=False, reason="categorical_missing_options")
        if ans not in opts:
            return ParsedQA(question=question, answer="", valid=False, reason="categorical_answer_not_in_options")
        return ParsedQA(question=question, answer=ans, valid=True, reason=None)


def extract_reasoner_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    boxed = extract_boxed_content(text)
    if boxed:
        if isinstance(boxed, list):
            boxed = boxed[-1] if boxed else ""
        ans = str(boxed).strip()
        if ans and ans.lower() != "none":
            return ans
    # fallback: sometimes models output just the answer
    return text.strip()
