from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from mathruler.grader import extract_boxed_content, grade_answer


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip(".")
    return s


def _looks_like_choice(gold: str) -> bool:
    g = _normalize_text(gold).upper()
    return g in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}


def _extract_boxed_answer_strict(text: str) -> Optional[str]:
    # Enforce that the model must output \boxed{...}; otherwise incorrect.
    boxed = extract_boxed_content(text)
    if not boxed:
        return None
    if isinstance(boxed, list):
        if not boxed:
            return None
        boxed = boxed[-1]
    ans = str(boxed).strip()
    if not ans:
        return None
    if ans.strip().lower() == "none":
        return None
    return ans


@dataclass
class AnswerVerifier:
    """Verifier aligned with `vllm_service_init/start_vllm_server_difficulty_aware.py`.

    - Requires prediction to contain a `\\boxed{...}` answer; otherwise returns False.
    - Uses `mathruler.grader.grade_answer` for equivalence checking.
    - Optionally applies a per-call timeout via `stopit` (same strategy as the server).
    """

    grade_timeout_s: float = 0.0  # 0 disables timeout

    def is_correct(self, pred_text: str, gold_answer: str) -> bool:
        gold = str(gold_answer or "").strip()
        if not pred_text or not gold:
            return False

        pred_boxed = _extract_boxed_answer_strict(str(pred_text))
        if pred_boxed is None or pred_boxed == "" or pred_boxed.lower() == "none":
            return False

        pred_n = _normalize_text(pred_boxed)
        gold_n = _normalize_text(gold)
        if not pred_n or not gold_n:
            return False

        # Cheap checks first (match server's categorical behavior expectations)
        if _looks_like_choice(gold_n):
            m = re.fullmatch(r"([A-J])", pred_n.upper())
            return bool(m and m.group(1) == gold_n.upper())
        if pred_n == gold_n:
            return True

        if self.grade_timeout_s and self.grade_timeout_s > 0:
            try:
                import stopit  # type: ignore

                @stopit.threading_timeoutable(default="TIMED_OUT")
                def _grade(res1, res2):
                    return grade_answer(res1, res2)

                out = _grade([pred_n], gold_n, timeout=float(self.grade_timeout_s))
                return bool(out) and out != "TIMED_OUT"
            except Exception:
                pass

        try:
            return bool(grade_answer([pred_n], gold_n))
        except Exception:
            return False
