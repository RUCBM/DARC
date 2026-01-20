"""Relatedness judging for SPICE challenger rewards."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional


def _get_env(*names: str) -> str:
    for name in names:
        if not name:
            continue
        val = os.getenv(name)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return ""


def _openai_judge_related(text: str, question: str) -> Optional[bool]:
    api_key = _get_env("API_KEY")
    model = _get_env("SPICE_REL_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
    base_url = _get_env("API_BASE_URL")

    if not api_key:
        print("[spice][related] API key not set; skip relation check (treat as related).")
        return None
    if not model:
        print("[spice][related] Model not set; skip relation check (treat as related).")
        return None

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"[spice][related] OpenAI SDK unavailable: {exc}")
        return None

    retries = int(_get_env("SPICE_REL_RETRIES", "DEEPSEEK_RETRIES") or "5")
    retry_wait = float(_get_env("SPICE_REL_RETRY_WAIT", "DEEPSEEK_RETRY_WAIT") or "5")
    http_timeout = float(_get_env("SPICE_REL_HTTP_TIMEOUT", "DEEPSEEK_HTTP_TIMEOUT") or "30")

    client_kwargs: Dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            client = OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise judge. Reply with yes or no only."},
                    {
                        "role": "user",
                        "content": (
                            "Do you think this question `" + (question or "") + "` is related to the following text:"
                            + (text or "")
                            + "\n\nReply with only one word: yes or no."
                        ),
                    },
                ],
                stream=False,
                temperature=0.0,
                timeout=http_timeout,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if not content:
                time.sleep(retry_wait)
                continue
            if ("yes" in content) and ("no" not in content):
                return True
            if ("no" in content) and ("yes" not in content):
                return False
            return content.startswith("y") and not content.startswith("n")
        except Exception as exc:
            last_exc = exc
            print(f"[spice][related] OpenAI call failed (attempt {attempt+1}/{retries}): {exc}")
            time.sleep(retry_wait)
    print(f"[spice][related] Judge failed after retries: {last_exc}")
    return None


def generate_relation_results(payload: List[Dict[str, str]]) -> List[Dict[str, float]]:
    if not payload:
        return []

    try:
        max_workers = int(_get_env("SPICE_REL_MAX_WORKERS", "DEEPSEEK_MAX_WORKERS") or "8")
    except Exception:
        max_workers = 8

    results: List[Optional[Dict[str, float]]] = [None] * len(payload)
    tasks: Dict[int, tuple[str, str]] = {}

    for i, item in enumerate(payload):
        text = item.get("text", "") or ""
        question = item.get("question", "") or ""
        if not text or not question:
            results[i] = {"text": text, "question": question, "related": False, "score": 0.0}
        else:
            tasks[i] = (text, question)

    if not tasks:
        return [r for r in results if r is not None]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_openai_judge_related, text, question): idx
            for idx, (text, question) in tasks.items()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            text, question = tasks[idx]
            try:
                flag = future.result()
            except Exception as exc:
                print(f"[spice][related] Future error at idx={idx}: {exc}")
                flag = None
            if flag is None:
                flag = True
            results[idx] = {
                "text": text,
                "question": question,
                "related": bool(flag),
                "score": 1.0 if flag else 0.0,
            }

    for i, r in enumerate(results):
        if r is None:
            item = payload[i]
            results[i] = {
                "text": item.get("text", "") or "",
                "question": item.get("question", "") or "",
                "related": True,
                "score": 1.0,
            }
    return results  # type: ignore
