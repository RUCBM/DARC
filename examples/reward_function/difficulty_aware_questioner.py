 # Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.                                                    
 
                                                                
from __future__ import annotations

import ast
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import regex as re
import requests
from openai import OpenAI                                  
from mathruler.grader import extract_boxed_content

STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    raise ValueError("STORAGE_PATH is required.")
SERVER_BASE_PORT = int(os.getenv("SOLVER_SERVER_BASE_PORT", "5000"))
SERVER_COUNT = int(os.getenv("SOLVER_SERVER_COUNT", "4"))
HTTP_TIMEOUT = float(os.getenv("SOLVER_HTTP_TIMEOUT", "1200"))
HTTP_RETRIES = int(os.getenv("SOLVER_HTTP_RETRIES", "2"))

                                                                      
ENABLE_COPY_PENALTY = os.getenv("ENABLE_COPY_PENALTY", "1") == "1"
COPY_N = int(os.getenv("COPY_N", "6"))                    
COPY_TEXT_MAXLEN = int(os.getenv("COPY_TEXT_MAXLEN", "50000"))                                    
COPY_HARD_THRESHOLD = float(os.getenv("COPY_HARD_THRESHOLD", "0.8"))                                             
def _parse_target_solver_accuracy() -> Dict[int, float]:





    default = {3: 0.2, 2: 0.5, 1: 0.8}
    env_val = os.getenv("TARGET_SOLVER_ACCURACY", "").strip()
    if not env_val:
        return default
    try:
        pairs = [p for p in env_val.split(",") if p]
        out: Dict[int, float] = {}
        for pair in pairs:
            k, v = pair.split(":", 1)
            out[int(k.strip())] = float(v.strip())
        if not out:
            return default
        return out
    except Exception:
        return default


TARGET_SOLVER_ACCURACY = _parse_target_solver_accuracy()
INVALID_SCORE = {"overall": -1.0, "format": 0.0, "accuracy": 0.0}
QUESTION_PATTERN = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_JSON = re.compile(r"^\s*```(?:json)?\s*(\{.*\})\s*```\s*$", re.IGNORECASE | re.DOTALL)
EXPRESSION_PATTERN = re.compile(r"[A-Za-z0-9_+\-*/^().= \\{}]+$")
                                                                  
_CATEGORICAL_OPTION_RE = re.compile(
    r"(?i)(?:^|\\r\\n|\\n|\\r|\r\n|\n|\r)\s*([A-J])\s*"
    r"(?:[\.\):：\uFF0E\uFF09\u3001])\s*"
)
_LEADING_ROLE_PREFIX = re.compile(r"^\s*(assistant)\s*[:：]\s*", re.IGNORECASE)

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


def split_list(lst: List[Dict[str, str]], n: int) -> List[List[Dict[str, str]]]:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def generate_temp_filename(prefix: str = "temp", suffix: str = ".json") -> str:
    temp_dir = os.path.join(STORAGE_PATH, "temp_results")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return os.path.join(temp_dir, f"{prefix}_{timestamp}_{rand_part}{suffix}")


def fetch(port: int, task_file: str) -> bool:
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = requests.get(
                f"http://0.0.0.0:{port}/hello",
                params={"name": task_file},
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            last_exc = exc
            print(f"[reward] Fetch attempt {attempt+1}/{HTTP_RETRIES+1} failed on port {port} for {task_file}: {exc}")
            time.sleep(min(5 * (attempt + 1), 30))
    print(f"[reward] Giving up on port {port} for {task_file}: {last_exc}")
    return False


def generate_results(payload: List[Dict[str, str]]) -> List[Dict[str, float]]:
    if not payload:
        return []

    shards = split_list(payload, SERVER_COUNT)
    temp_files = [generate_temp_filename(prefix=f"temp_{i}") for i in range(SERVER_COUNT)]
    shard_lens = [len(s) for s in shards]

    for path, shard in zip(temp_files, shards):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(shard, fp, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=SERVER_COUNT) as executor:
        futures = [
            executor.submit(fetch, SERVER_BASE_PORT + idx, task_file)
            for idx, task_file in enumerate(temp_files)
        ]
        for idx, future in enumerate(as_completed(futures)):
            ok = future.result()
            if not ok:
                print(f"[reward] Shard {idx} request did not succeed; will fill with placeholders if needed.")

    final_results: List[Dict[str, float]] = []
    for shard_len, path in zip(shard_lens, temp_files):
        result_path = path.replace(".json", "_results.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as fp:
                    final_results.extend(json.load(fp))
            finally:
                os.remove(result_path)
        else:
                                                                          
            final_results.extend([
                {"question": "", "answer": "", "score": -1.0}
                for _ in range(shard_len)
            ])

    return final_results


def generate_answers_from_text(payload: List[Dict[str, str]]) -> List[Dict[str, float]]:






    if not payload:
        return []

    shards = split_list(payload, SERVER_COUNT)
    temp_files = [generate_temp_filename(prefix=f"answer_{i}") for i in range(SERVER_COUNT)]
    shard_lens = [len(s) for s in shards]

    for path, shard in zip(temp_files, shards):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(shard, fp, ensure_ascii=False)

    def _fetch_answer(port: int, task_file: str) -> bool:
        last_exc: Optional[Exception] = None
        for attempt in range(HTTP_RETRIES + 1):
            try:
                resp = requests.get(
                    f"http://0.0.0.0:{port}/answer",
                    params={"name": task_file},
                    timeout=HTTP_TIMEOUT,
                )
                resp.raise_for_status()
                return True
            except Exception as exc:
                last_exc = exc
                print(f"[answer] Fetch attempt {attempt+1}/{HTTP_RETRIES+1} failed on port {port} for {task_file}: {exc}")
                time.sleep(min(5 * (attempt + 1), 30))
        print(f"[answer] Giving up on port {port} for {task_file}: {last_exc}")
        return False

    with ThreadPoolExecutor(max_workers=SERVER_COUNT) as executor:
        futures = [
            executor.submit(_fetch_answer, SERVER_BASE_PORT + idx, task_file)
            for idx, task_file in enumerate(temp_files)
        ]
        for idx, future in enumerate(as_completed(futures)):
            ok = future.result()
            if not ok:
                print(f"[answer] Shard {idx} request did not succeed; will fill with placeholders if needed.")

    final_results: List[Dict[str, float]] = []
    for shard_len, path in zip(shard_lens, temp_files):
        result_path = path.replace(".json", "_results.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as fp:
                    final_results.extend(json.load(fp))
            finally:
                os.remove(result_path)
        else:
            final_results.extend([
                {"text": "", "question": "", "answer": "", "majority_fraction": 0.0}
                for _ in range(shard_len)
            ])

    return final_results


def extract_question_and_answer(predict: str) -> Tuple[Optional[str], Optional[str]]:













    if not isinstance(predict, str):
        print("[format] Invalid predict type; expected str.")
        return None, None

    stripped = predict.strip()
                                                                                 
                                                                      
    stripped2 = _LEADING_ROLE_PREFIX.sub("", stripped, count=1)
    if stripped2 != stripped:
                                                                    
        head = stripped2.lstrip()
        if head.startswith("```") or head.startswith("{"):
            stripped = stripped2

                                                   
    m = CODE_FENCE_JSON.match(stripped)
    if m:
        json_text = m.group(1).strip()
    else:
        if not (stripped.startswith("{") and stripped.endswith("}")):
            print("[format] Output is not a single JSON object or fenced JSON block.")
            return None, None
        json_text = stripped

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"[format] JSON parse error: {e}")
        return None, None

    required_keys = {
        "analysis",
        "question",
        "intermediate_results",
        "answer",
        "solving_time_estimate",
        "required_concepts",
        "potential_errors",
    }
    if set(payload.keys()) != required_keys:
        print(f"[format] Invalid JSON keys. got={sorted(list(payload.keys()))}, expected={sorted(list(required_keys))}")
        return None, None

    question = payload.get("question")
    answer_val = payload.get("answer")

    if not isinstance(question, str) or not question.strip():
        print("[format] 'question' must be a non-empty string.")
        return None, None

                                                                                     
                                        
    if not isinstance(answer_val, (int, float, str)):
        print("[format] 'answer' must be a JSON int/float/string value.")
        return None, None

    return question.strip(), str(answer_val)


def extract_question_only(predict: str) -> Optional[str]:





    if not isinstance(predict, str):
        print("[format] Invalid predict type; expected str.")
        return None

    stripped = predict.strip()
    stripped2 = _LEADING_ROLE_PREFIX.sub("", stripped, count=1)
    if stripped2 != stripped:
        head = stripped2.lstrip()
        if head.startswith("```") or head.startswith("{"):
            stripped = stripped2
    m = CODE_FENCE_JSON.match(stripped)
    if m:
        json_text = m.group(1).strip()
    else:
        if not (stripped.startswith("{") and stripped.endswith("}")):
            print("[format] Output is not a single JSON object or fenced JSON block.")
            return None
        json_text = stripped

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"[format] JSON parse error: {e}")
        return None

    required_keys = {
        "analysis",
        "question",
        "intermediate_results",
        "answer",
        "solving_time_estimate",
        "required_concepts",
        "potential_errors",
    }
    if set(payload.keys()) != required_keys:
        print(f"[format] Invalid JSON keys. got={sorted(list(payload.keys()))}, expected={sorted(list(required_keys))}")
        return None

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        print("[format] 'question' must be a non-empty string.")
        return None
    return question.strip()


def _coerce_to_python(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_ground_truth(raw_ground_truth) -> Tuple[Optional[int], Optional[str]]:
    raw_ground_truth = _coerce_to_python(raw_ground_truth)
    difficulty_id: Optional[int] = None
    answer_type: Optional[str] = None

    def _assign_difficulty(value) -> None:
        nonlocal difficulty_id
        if difficulty_id is not None:
            return
        try:
            difficulty_id = int(value)
        except (TypeError, ValueError):
            pass

    def _assign_answer_type(value) -> None:
        nonlocal answer_type
        if answer_type is not None:
            return
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped:
                answer_type = stripped

    def _walk(obj) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            stripped = obj.strip()
            if not stripped:
                return
            try:
                _walk(json.loads(stripped))
                return
            except json.JSONDecodeError:
                try:
                    _walk(ast.literal_eval(stripped))
                    return
                except (ValueError, SyntaxError):
                    _assign_difficulty(stripped)
            return
        if isinstance(obj, (int, float)):
            _assign_difficulty(obj)
            return
        if isinstance(obj, dict):
            if "ground_truth" in obj:
                _walk(obj["ground_truth"])
            if "difficulty_id" in obj:
                _assign_difficulty(obj["difficulty_id"])
            if "difficulty" in obj:
                _assign_difficulty(obj["difficulty"])
            if "level" in obj:
                _assign_difficulty(obj["level"])
            if "answer_type" in obj:
                _assign_answer_type(obj["answer_type"])
            if "answer_format" in obj:
                _assign_answer_type(obj["answer_format"])
            if "expected_answer_type" in obj:
                _assign_answer_type(obj["expected_answer_type"])
            if "extra_info" in obj:
                _walk(obj["extra_info"])
            return
        if isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(raw_ground_truth)
    return difficulty_id, answer_type


def extract_text_from_ground_truth(obj) -> Optional[str]:



    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
    except Exception:
        pass

    def _walk(o):
        if o is None:
            return None
        if isinstance(o, dict):
            for k in ["text", "context", "passage", "source_text", "content"]:
                if k in o and isinstance(o[k], str) and o[k].strip():
                    return o[k]
            for k in ["extra_info", "ground_truth", "meta"]:
                if k in o:
                    v = _walk(o[k])
                    if v:
                        return v
            return None
        if isinstance(o, list):
            for it in o:
                v = _walk(it)
                if v:
                    return v
        return None

    return _walk(obj)


def validate_answer_type(answer: str, answer_type: Optional[str]) -> bool:
    if not answer_type:
        return True

    normalized = answer_type.lower()
    cleaned = answer.strip()
    if normalized == "integer":
        return bool(re.fullmatch(r"[-+]?\d+", cleaned))
    if normalized == "float":
        return bool(re.fullmatch(r"[-+]?\d+\.\d+", cleaned))
    if normalized == "categorical":
        return normalize_categorical_answer(cleaned) is not None
    if normalized == "string":
        if not re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){0,2}", cleaned):
            return False
        return 1 <= len(cleaned.split()) <= 3
    if normalized == "boolean":
        return cleaned.lower() in {"true", "false"}
    if normalized == "expression":
        return bool(cleaned and EXPRESSION_PATTERN.fullmatch(cleaned) and any(ch in cleaned for ch in "+-*/^="))

    return True


def normalize_categorical_answer(answer: str) -> Optional[str]:






    if not isinstance(answer, str):
        return None
    s = answer.strip()
    if not s:
        return None

    s = s.upper()
                                                                        
    for _ in range(3):
        s_new = re.sub(r"\\(?:TEXT|MATHRM|MATHBF|BM|BOLD|BOLDFACE|RM|BF)\s*\{\s*([A-J])\s*\}", r"\1", s)
        if s_new == s:
            break
        s = s_new

                          
    if re.fullmatch(r"[A-J]", s):
        return s
    if re.fullmatch(r"\(\s*[A-J]\s*\)", s):
        return re.sub(r"[()\s]", "", s)
    if re.fullmatch(r"[A-J][\.\)\:\,;]\s*", s):
        return s[0]

                                                   
    s = re.sub(r"[{}\[\]()<>\t\r\n]", " ", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

                                             
    hits = re.findall(r"\b([A-J])\b", s)
    distinct = sorted(set(hits))
    if len(distinct) == 1:
        return distinct[0]
    return None


def categorical_question_has_options(question: str, *, min_distinct: int = 3) -> bool:





    if not isinstance(question, str) or not question.strip():
        return False
    hits = _CATEGORICAL_OPTION_RE.findall(question.upper())
    return len(set(hits)) >= int(min_distinct)


def difficulty_reward(solver_score: Optional[float], difficulty_id: Optional[int]) -> float:





    if solver_score is None or solver_score < 0:
        return -1.0
    if difficulty_id not in TARGET_SOLVER_ACCURACY:
        return -1.0
    if solver_score == 0:
        return -0.1

    target = float(TARGET_SOLVER_ACCURACY[difficulty_id])

    score = 1 - abs(solver_score - target)
    return score


def _normalize_for_ngram(s: str) -> str:







    if not isinstance(s, str):
        return ""
    s = s.lower()
                                                        
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _char_ngrams(s: str, n: int) -> List[str]:
    if n <= 0:
        return []
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]


def compute_copy_ratio(text: str, question: str, n: int = COPY_N, maxlen: int = COPY_TEXT_MAXLEN) -> float:






    try:
        if not text or not question:
            return 0.0
        qn = _normalize_for_ngram(question)
        tn_raw = text[: maxlen]
        tn = _normalize_for_ngram(tn_raw)
        if not qn or not tn:
            return 0.0
                                                   
        if qn and qn in tn:
            return 1.0
                                         
        n = max(2, int(n))
        q_ngrams = _char_ngrams(qn.replace(" ", ""), n)
        if not q_ngrams:
                                                                   
            q_tokens = [t for t in qn.split() if t]
            if not q_tokens:
                return 0.0
            t_set = set(tn.split())
            overlap = sum(1 for t in q_tokens if t in t_set)
            return overlap / max(1, len(q_tokens))
        t_ngrams = set(_char_ngrams(tn.replace(" ", ""), n))
        if not t_ngrams:
            return 0.0
        hit = sum(1 for g in q_ngrams if g in t_ngrams)
        return hit / max(1, len(q_ngrams))
    except Exception:
        return 0.0


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_RETRIES = int(os.getenv("DEEPSEEK_RETRIES", "5"))
DEEPSEEK_RETRY_WAIT = float(os.getenv("DEEPSEEK_RETRY_WAIT", "5"))
DEEPSEEK_HTTP_TIMEOUT = float(os.getenv("DEEPSEEK_HTTP_TIMEOUT", "30"))


def _deepseek_judge_related(text: str, question: str) -> Optional[bool]:
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        print("[related] API_KEY is not set; skip relation check (treat as related).")
        return None

    t = text if text is not None else ""

    last_exc: Optional[Exception] = None
    for attempt in range(DEEPSEEK_RETRIES):
        try:
            client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise judge. Reply with yes or no only."},
                    {
                        "role": "user",
                        "content": (
                            "Do you think this question `" + (question or "") + "` is related to the following text:"\
                            + t + "\n\nReply with only one word: yes or no."
                        ),
                    },
                ],
                stream=False,
                temperature=0.0,
                timeout=DEEPSEEK_HTTP_TIMEOUT,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if not content:
                time.sleep(DEEPSEEK_RETRY_WAIT)
                continue
            if ("yes" in content) and ("no" not in content):
                return True
            if ("no" in content) and ("yes" not in content):
                return False
            return content.startswith("y") and not content.startswith("n")
        except Exception as exc:
            last_exc = exc
            print(f"[related] DeepSeek(OpenAI SDK) call failed (attempt {attempt+1}/{DEEPSEEK_RETRIES}): {exc}")
            time.sleep(DEEPSEEK_RETRY_WAIT)
    print(f"[related] DeepSeek judge failed after retries: {last_exc}")
    return None


def generate_relation_results(payload: List[Dict[str, str]]) -> List[Dict[str, float]]:
    if not payload:
        return []

    max_workers = int(os.getenv("DEEPSEEK_MAX_WORKERS", "8"))
                                                               
    results: List[Optional[Dict[str, float]]] = [None] * len(payload)
    tasks: Dict = {}

                                                                 
    for i, item in enumerate(payload):
        text = item.get("text", "") or ""
        question = item.get("question", "") or ""
        if not text or not question:
            results[i] = {"text": text, "question": question, "related": False, "score": 0.0}
        else:
            tasks[i] = (text, question)

    if not tasks:
                           
        return [r for r in results if r is not None]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_deepseek_judge_related, text, question): idx
            for idx, (text, question) in tasks.items()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            text, question = tasks[idx]
            try:
                flag = future.result()
            except Exception as exc:
                print(f"[related] Future error at idx={idx}: {exc}")
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
    return results                


def relative_reward(text: str, question: str) -> float:

    if not text or not question:
        return 0.0
    results = generate_relation_results([{"text": text, "question": question}])
    if not results:
        return 0.0
    return 1.0 if results[0].get("related") else -1.0


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.0,
    file_path: str = "",
) -> List[Dict[str, float]]:
    scores: List[Optional[Dict[str, float]]] = [INVALID_SCORE.copy() for _ in predicts]
    solver_payload: List[Dict[str, str]] = []
    solver_meta: List[Dict[str, int]] = []

    use_text_for_answer = os.getenv("USE_TEXT_SOLVER_FOR_ANSWER", "1") == "1"
    answer_payload: List[Dict[str, str]] = []
    answer_meta: List[Dict[str, object]] = []                                     

    for idx, predict in enumerate(predicts):
        print(f"predict: {predict}")
        if use_text_for_answer:
            question = extract_question_only(predict)
            if question is None:
                continue
            difficulty_id, answer_type = parse_ground_truth(ground_truths[idx])
            if difficulty_id is None:
                continue
            if isinstance(answer_type, str) and answer_type.lower() == "categorical":
                if not categorical_question_has_options(question):
                    scores[idx] = INVALID_SCORE.copy()
                    continue
                                                                              
            gt_text = extract_text_from_ground_truth(ground_truths[idx]) or ""
            answer_payload.append({"text": gt_text, "question": question, "answer_type": answer_type})
            answer_meta.append({"index": idx, "difficulty_id": difficulty_id, "answer_type": answer_type, "question": question, "text": gt_text})
            scores[idx] = None                       
        else:
            question, answer = extract_question_and_answer(predict)
            if question is None or answer is None:
                continue

            difficulty_id, answer_type = parse_ground_truth(ground_truths[idx])
            if answer_type and isinstance(answer_type, str) and answer_type.lower() == "categorical":
                if not categorical_question_has_options(question):
                    continue
                answer = normalize_categorical_answer(answer) or answer
            if answer_type and not validate_answer_type(answer, answer_type):
                continue
            if difficulty_id is None:
                continue

                                                                                   
            normalized_answer = answer
            if answer_type and isinstance(answer_type, str):
                at = answer_type.lower()
                if at == "boolean":
                    normalized_answer = answer.lower()

            solver_payload.append({"question": question, "answer": normalized_answer, "answer_type": answer_type})
            solver_meta.append({"index": idx, "difficulty_id": difficulty_id})
            scores[idx] = None                       

                                                                    
    text_answers = []
    if use_text_for_answer and answer_payload:
        text_answers = generate_answers_from_text(answer_payload)
                                                                                 
        for meta, ans in zip(answer_meta, text_answers):
            idx = int(meta["index"])                
            did = int(meta["difficulty_id"])                
            atype = meta.get("answer_type")                 
            maj_ans = (ans.get("answer") or "").strip() if isinstance(ans, dict) else ""
            if isinstance(atype, str) and atype.lower() == "categorical":
                maj_ans = normalize_categorical_answer(maj_ans) or maj_ans
                                                            
            if atype and not validate_answer_type(maj_ans, atype):
                                     
                continue
                                               
            normalized_answer = maj_ans
            if isinstance(atype, str) and atype.lower() == "boolean":
                normalized_answer = maj_ans.lower()
            solver_payload.append({
                "question": str(meta.get("question", "")),
                "answer": normalized_answer,
                "answer_type": atype,
            })
            solver_meta.append({"index": idx, "difficulty_id": did})

    solver_results = generate_results(solver_payload) if solver_payload else []

                                                                                      
                                                                                    
    for i, s in enumerate(scores):
        if s is None:
            scores[i] = INVALID_SCORE.copy()

                             
    relation_payload: List[Dict[str, str]] = []
    relation_meta: List[int] = []
                                                                                      

    def _extract_doc_id(obj) -> Optional[int]:

        try:
            if isinstance(obj, str):
                obj = json.loads(obj)
        except Exception:
            pass
        def _walk(o):
            if o is None:
                return None
            if isinstance(o, dict):
                for k in ["doc_id", "index", "id"]:
                    if k in o:
                        try:
                            return int(o[k])
                        except Exception:
                            continue
                for k in ["extra_info", "meta", "ground_truth"]:
                    if k in o:
                        v = _walk(o[k])
                        if v is not None:
                            return v
                return None
            if isinstance(o, list):
                for it in o:
                    v = _walk(it)
                    if v is not None:
                        return v
            return None
        return _walk(obj)

                                           
    diff_count: Dict[int, int] = {}
    diff_solver_sum: Dict[int, float] = {}
    diff_reward_sum: Dict[int, float] = {}
                                                 
    debug_records: List[Dict[str, object]] = []

    for meta, payload, solver_res, predict in zip(solver_meta, solver_payload, solver_results, predicts):
        idx = meta["index"]
        solver_score = solver_res.get("score", -1)
        reward = difficulty_reward(solver_score, meta["difficulty_id"])
        scores[idx] = {
            "overall": reward,
            "format": 1.0,
            "accuracy": float(solver_score) if isinstance(solver_score, (int, float)) else 0.0,
        }
                                                                 
        debug_rec = {
            "index": idx,
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "solver_score": solver_score,
            "difficulty_id": meta.get("difficulty_id"),
            "reward": reward,
        }
                                                                                    
        if use_text_for_answer and answer_meta:
                                                         
            try:
                                                      
                for am, an in zip(answer_meta, text_answers):
                    if int(am["index"]) == idx:
                        debug_rec["text_majority_fraction"] = float(an.get("majority_fraction", 0.0)) if isinstance(an, dict) else 0.0
                        break
            except Exception:
                pass
        debug_records.append(debug_rec)

                                                         
        gt_text = extract_text_from_ground_truth(ground_truths[idx])
                                                                              
        if ENABLE_COPY_PENALTY and gt_text and payload.get("question"):
            try:
                cr = compute_copy_ratio(gt_text, payload.get("question", ""))
                debug_rec["copy_ratio"] = float(cr)
                if cr > COPY_HARD_THRESHOLD:
                    scores[idx]["overall"] = -1.0
            except Exception:
                pass
        if gt_text and payload.get("question"):
            relation_payload.append({"text": gt_text, "question": payload["question"]})
            relation_meta.append(idx)

                                  
        did = int(meta["difficulty_id"])
        diff_count[did] = diff_count.get(did, 0) + 1
        try:
            ss = float(solver_score)
        except Exception:
            ss = 0.0
        rr = float(reward) if isinstance(reward, (int, float)) else 0.0
        diff_solver_sum[did] = diff_solver_sum.get(did, 0.0) + ss
        diff_reward_sum[did] = diff_reward_sum.get(did, 0.0) + rr

                                                 
    for meta in solver_meta[len(solver_results):]:
        idx = meta["index"]
        scores[idx] = INVALID_SCORE.copy()

                                            
    for idx, score in enumerate(scores):
        if score is None:
            scores[idx] = INVALID_SCORE.copy()

                                                                    
    if relation_payload:
        relation_results = generate_relation_results(relation_payload)
        relation_flags: Dict[int, bool] = {}
        for meta_idx, rel in zip(relation_meta, relation_results):
            is_related = bool(rel.get("related", False))
            relation_flags[meta_idx] = is_related
            if not is_related:
                scores[meta_idx]["overall"] = -1.0
    else:
        relation_flags = {}

                                                                              
                                                                              
    if os.getenv("ENABLE_DIFFICULTY_RANKING", "1") == "1":
        margin = float(os.getenv("RANKING_MARGIN", "0.2"))
        weight = float(os.getenv("RANKING_WEIGHT", "0.5"))

                                                                                
        groups: Dict[int, List[Tuple[int, int, float]]] = {}
        for meta, solver_res in zip(solver_meta, solver_results):
            i = meta["index"]
            if not isinstance(scores[i], dict):
                continue
            doc_id = _extract_doc_id(ground_truths[i])
            if doc_id is None:
                continue
            did = int(meta["difficulty_id"])
            try:
                ss = float(solver_res.get("score", -1))
            except Exception:
                ss = -1.0
            if ss < 0 or scores[i]['overall'] < 0:
                continue
            groups.setdefault(doc_id, []).append((i, did, ss))

                                                           
        for _, items in groups.items():
            items = [(i, d, s) for (i, d, s) in items if 1 <= d <= 5]
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[1])                          
                                                   
            for a, b in zip(items, items[1:]):
                idx_lo, d_lo, s_lo = a
                idx_hi, d_hi, s_hi = b
                if d_hi <= d_lo:
                    continue
                penalty = max(0.0, (s_hi + margin) - s_lo)
                if penalty > 0:
                                                    
                    scores[idx_lo]["overall"] = max(0, scores[idx_lo]["overall"] - weight * penalty)
                    scores[idx_hi]["overall"] = max(0, scores[idx_hi]["overall"] - weight * penalty)

                                                                                                          
                                                                     
    if diff_count:
                                                               
        diff_overall_sum: Dict[int, float] = {}
        for meta in solver_meta:
            i = meta["index"]
            if not isinstance(scores[i], dict):
                continue
            did = int(meta["difficulty_id"])
            try:
                ov = float(scores[i].get("overall", 0.0))
            except Exception:
                ov = 0.0
            diff_overall_sum[did] = diff_overall_sum.get(did, 0.0) + ov

        for idx in range(len(scores)):
            for did, cnt in diff_count.items():
                if cnt <= 0:
                    continue                                          
                solver_mean = diff_solver_sum[did] / cnt
                reward_mean = diff_reward_sum[did] / cnt
                overall_mean = diff_overall_sum.get(did, 0.0) / cnt
                scores[idx][f"by_difficulty/{did}/solver_score_mean"] = float(solver_mean)
                scores[idx][f"by_difficulty/{did}/reward_mean"] = float(reward_mean)
                scores[idx][f"by_difficulty/{did}/overall_mean"] = float(overall_mean)
                scores[idx][f"by_difficulty/{did}/count"] = float(cnt)

                                                                   
    for rec in debug_records:
        ridx = rec["index"]
        is_related = relation_flags.get(ridx, None)
        overall = scores[ridx]["overall"] if 0 <= ridx < len(scores) and isinstance(scores[ridx], dict) else None
        extra = ''
        if 'text_majority_fraction' in rec:
            try:
                extra = f", text_majority_fraction: {float(rec['text_majority_fraction']):.3f}"
            except Exception:
                extra = f", text_majority_fraction: {rec['text_majority_fraction']}"
        if 'copy_ratio' in rec:
            try:
                extra += f", copy_ratio: {float(rec['copy_ratio']):.3f}"
            except Exception:
                extra += f", copy_ratio: {rec['copy_ratio']}"
        print(
            f"question: {rec['question']}, answer: {rec['answer']}, "
            f"solver_score: {rec['solver_score']}, difficulty_id:{rec['difficulty_id']}, "
            f"difficulty reward: {rec['reward']}, overall: {overall}, related: {is_related}{extra}"
        )
    return scores
