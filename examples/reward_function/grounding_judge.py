"""Qwen3 Base grounding experiment, using vLLM 0.8.5 guided decoding."""

import os
import time


class GroundingJudgeError(RuntimeError):
    """No valid grounding verdict was obtained; do not turn this into a reward."""


def judge_related(text: str, question: str) -> bool:
    from openai import OpenAI

    attempts = int(os.getenv("GROUNDING_JUDGE_RETRIES", "3"))
    if attempts < 1:
        raise ValueError("GROUNDING_JUDGE_RETRIES must be at least 1")
    base_url = os.getenv("GROUNDING_JUDGE_BASE_URL", "http://127.0.0.1:6100/v1")
    model = os.getenv("GROUNDING_JUDGE_MODEL", "Qwen3-4B-Base")
    # EMPTY is an SDK placeholder for a local server with authentication disabled.
    api_key = os.getenv("GROUNDING_JUDGE_API_KEY", "").strip() or "EMPTY"
    timeout = float(os.getenv("GROUNDING_JUDGE_TIMEOUT", "60"))
    # Preserve the baseline relevance question; use a completion for the Base model.
    prompt = (
        "You are a precise judge. Reply with yes or no only.\n\n"
        "Do you think this question `" + question
        + "` is related to the following text:" + text
        + "\n\nReply with only one word: yes or no.\nAnswer:"
    )
    last_error_type = "unknown"
    with OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0) as client:
        for attempt in range(attempts):
            try:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8,
                    n=1,
                    stream=False,
                    extra_body={"guided_choice": ["yes", "no"]},
                )
                choice = response.choices[0]
                # Check the entire output, including whitespace, and reject truncation.
                if choice.finish_reason != "stop" or choice.text not in ("yes", "no"):
                    raise GroundingJudgeError("Expected a complete constrained yes/no output")
                return choice.text == "yes"
            except Exception as exc:
                last_error_type = type(exc).__name__
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 8))
    # Do not expose request bodies, documents, or credentials in error logs.
    raise GroundingJudgeError(
        f"Qwen3 grounding judge failed after {attempts} attempts ({last_error_type}); "
        "no grounding verdict was recorded."
    ) from None


if __name__ == "__main__":
    verdict = judge_related("Paris is the capital of France.", "What is the capital of France?")
    print("Qwen3 grounding preflight completed: " + ("yes" if verdict else "no"))
