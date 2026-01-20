import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from collections import Counter
import os
import re
import random

CATEGORICAL_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer option within \\boxed{}."
    " Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
)

                                                                  
_CATEGORICAL_OPTION_RE = re.compile(
    r"(?i)(?:^|\\r\\n|\\n|\\r|\r\n|\n|\r)\s*([A-J])\s*"
    r"(?:[\.\):：\uFF0E\uFF09\u3001])\s*"
)


def categorical_question_has_options(question: str, *, min_distinct: int = 3) -> bool:
    if not isinstance(question, str) or not question.strip():
        return False
    hits = _CATEGORICAL_OPTION_RE.findall(question.upper())
    return len(set(hits)) >= int(min_distinct)


def _normalize_answer_type(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _require_answer_type(sample: dict) -> str:
    at = _normalize_answer_type(sample.get("answer_type"))
    if not at:
        raise ValueError("Missing required field `answer_type` in sample.")
    return at


def _chat_to_prompt_fallback(chat: list[dict]) -> str:

    messages = list(chat) if chat else []
    last_role = (messages[-1].get("role") or "user").strip() if messages else ""
    if last_role != "assistant":
        messages.append({"role": "assistant", "content": ""})
    lines = []
    for msg in messages:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _tokenizer_has_chat_template(tokenizer_path: str) -> bool:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return bool(getattr(tokenizer, "chat_template", None))


def process_one(idx, messages, client: OpenAI, model_name: str, use_chat_template: bool = True):





    try:
                                                
        if use_chat_template:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0
            )
            content = resp.choices[0].message.content
        else:
            prompt = _chat_to_prompt_fallback(messages)
            resp = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=1.0
            )
            content = resp.choices[0].text
                                   
        decision = {
            "status": "ok",
            "raw": content,
        }
        accepted = True                   
        return idx, accepted, decision
    except Exception as e:
        decision = {
            "status": "error",
            "raw": str(e),
        }
        return idx, False, decision


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _gen_single_item(idx, sample):
                               
    assert 'text' in sample
    assert 'answer' in sample
    assert len(sample['text']) > 0

    assert isinstance(sample, dict)
    for k in ["text", "question", "answer"]:
        assert k in sample, f"missing key: {k}"

    assert isinstance(sample["text"], str),\
        f"text must be str, got {type(sample['text'])}"

    if isinstance(sample["question"], list) or isinstance(sample["question"], tuple):
        assert len(sample["question"]) == 1
        sample["question"] = sample["question"][0]

    assert isinstance(sample["question"], str),\
        f"question must be str, got {type(sample['question'])}"

    answer_type = _require_answer_type(sample)

    if answer_type == "categorical":
        prompt = [
            {"role": "user", "content": sample["question"] + CATEGORICAL_INSTRUCTION},
        ]
        text_prompt = [
            {
                "role": "user",
                "content": (
                    "Read the following context and answer the question.\n\n"
                    f"Context:\n{sample['text']}\n\n"
                    f"Question: {sample['question']}"
                    + CATEGORICAL_INSTRUCTION
                ),
            },
        ]
    else:
        prompt = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": sample["question"]},
        ]
        text_prompt = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {
                "role": "user",
                "content": (
                    "Read the following context and answer the question.\n\n"
                    f"Context:\n{sample['text']}\n\n"
                    f"Question: {sample['question']}"
                ),
            },
        ]

    return {
        "data_source": "solver",
        "prompt": prompt,
        "text_prompt": text_prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(sample['answer'])},
        "extra_info": {
            "split": "train",
            "index": idx,
            "text": sample['text'],
                                                  
            'difficulty_id': sample['difficulty_id'],
            "answer_type": answer_type,
                                                                      
        },
    }

def has_surrogate_in_anything(x) -> bool:
                                              
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in x)
    if isinstance(x, (list, tuple)):
        return any(has_surrogate_in_anything(v) for v in x)
    if isinstance(x, dict):
        return any(has_surrogate_in_anything(k) or has_surrogate_in_anything(v) for k, v in x.items())
                                       
    try:
        s = str(x)
        return any(0xD800 <= ord(c) <= 0xDFFF for c in s)
    except Exception:
        return False


import time
import json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


SYSTEM_PROMPT = """You are a dataset filtering judge.

Decide whether to KEEP the sample.

KEEP if the TEXT clearly helps answer the QUESTION and the QUESTION can be answered without the document.

Be lenient. Do NOT over-filter.

Return ONLY one word:
true or false
"""


def _call_judge(client, model, question, text, max_retries=3):
    user_prompt = f"""QUESTION:
{question}

TEXT:
{text}

Should this sample be kept?"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            out = resp.choices[0].message.content.strip().lower()
            if "true" in out:
                return True
            if "false" in out:
                return False
        except Exception:
            time.sleep(0.8 * (2 ** attempt))

                    
    return True

def assert_curriculum_order(df):
    did = df["extra_info"].apply(lambda x: x.get("difficulty_id", -1)).tolist()
    idxs = df["extra_info"].apply(lambda x: x.get("index", -1)).tolist()
    assert idxs == sorted(idxs), "extra_info.index is not non-decreasing (order corrupted?)"
                                       
    assert did == sorted(did), "difficulty_id is not non-decreasing (order corrupted?)"


def filter_unrelative(df, base_url, api_key, model="gpt-5-mini", max_workers=32):
    client = OpenAI(api_key=api_key, base_url=base_url)

    data = df.to_dict(orient="records")

    questions = [sample["prompt"][1]["content"] for sample in data]
    texts = [sample["extra_info"]["text"] for sample in data]

    keep_mask = [True] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_judge, client, model, questions[i], texts[i]
            ): i
            for i in range(len(data))
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Filtering (LLM judge)",
        ):
            idx = futures[future]
            try:
                keep_mask[idx] = future.result()
            except Exception:
                keep_mask[idx] = True             

    print("keep ratio:", keep_mask.count(True) / len(keep_mask))
    return df.loc[keep_mask].reset_index(drop=True)

from collections import Counter, defaultdict
import numpy as np
def analyze_answer_type_balance(
    answer_types,
    batch_size=128,
    drop_last=True,
):



                
    batches = []
    n = len(answer_types)
    for i in range(0, n, batch_size):
        if drop_last and i + batch_size > n:
            break
        batches.append(answer_types[i:i + batch_size])

    num_batches = len(batches)
    print(f"Total samples: {n}")
    print(f"Batch size: {batch_size}")
    print(f"Num batches: {num_batches}")

                             
    all_types = set(answer_types)

                                       
    type_to_props = defaultdict(list)

    for batch in batches:
        cnt = Counter(batch)
        for t in all_types:
            prop = cnt.get(t, 0) / batch_size
            type_to_props[t].append(prop)

               
    stats = {}
    for t, props in type_to_props.items():
        arr = np.array(props, dtype=float)
        stats[t] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "var": float(arr.var()),
        }

    return stats



def filter_duplicate(df, model_path):
                               
               
                               
    data = df.to_dict(orient="records")

    questions = [
        sample["prompt"][1]["content"]
        for sample in data
    ]

                               
                
                               
    device = "cpu"
    model = SentenceTransformer(model_path, device=device)

                               
                          
                               
    embeddings = model.encode(
        questions,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,                  
        show_progress_bar=True
    )

                               
                                                
                               
    threshold = 0.6

    kept_indices = []
    kept_embeddings = []

               
    for idx in tqdm(range(len(embeddings) - 1, -1, -1)):
        emb = embeddings[idx]

        if len(kept_embeddings) == 0:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
            continue

        sims = util.cos_sim(emb, torch.stack(kept_embeddings))[0]
        max_sim = sims.max().item()

        if max_sim < threshold:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
                                         

                               
                                             
                               
    kept_indices = sorted(kept_indices)

    filtered_data = [data[i] for i in kept_indices]

    print(f"Original size: {len(data)}")
    print(f"Filtered size: {len(filtered_data)}")
    print(f"Removed: {len(data) - len(filtered_data)}")

    return pd.DataFrame(filtered_data)


                                                                                                                   
                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="questioner")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:6000/v1")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="如果提供且没有 chat_template，则回退到纯文本 prompt。",
    )

    parser.add_argument("--judge_api_key", type=str, default=os.getenv("API_KEY"))
    parser.add_argument("--judge_base_url", type=str, default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.getenv("QUESTIONER_SAVE_PATH"),
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--solver_save_path",
        type=str,
        default=os.getenv("SOLVER_SAVE_PATH"),
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--sentence_transformer_path",
        type=str,
        default=os.getenv("SENTENCE_TRANSFORMER_PATH"),
        help="SentenceTransformer 模型路径",
    )

    parser.add_argument(
        "--dup_solver_save_path",
        type=str,
        default=os.getenv("DUP_SOLVER_SAVE_PATH"),
        help="输出 parquet 路径",
    )

    parser.add_argument(
        "--parquet_path",
        type=str,
        default=os.getenv("PARQUET_PATH"),
        help="输入 parquet 路径",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="并发线程数",
    )

    args = parser.parse_args()
    required_paths = {
        "save_path": "QUESTIONER_SAVE_PATH",
        "solver_save_path": "SOLVER_SAVE_PATH",
        "dup_solver_save_path": "DUP_SOLVER_SAVE_PATH",
        "parquet_path": "PARQUET_PATH",
    }
    missing = [env for arg, env in required_paths.items() if not getattr(args, arg)]
    if missing:
        raise ValueError("Missing required paths: " + ", ".join(missing))
    use_chat_template = True
    if args.tokenizer_path:
        use_chat_template = _tokenizer_has_chat_template(args.tokenizer_path)

                      
    ensure_parent_dir(args.save_path)
    ensure_parent_dir(args.solver_save_path)
    ensure_parent_dir(args.dup_solver_save_path)

                                               
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

               
    df = pd.read_parquet(args.parquet_path)


                       
    if "prompt" not in df.columns:
        raise ValueError(
            f"'prompt' column not found in dataframe columns: {df.columns.tolist()}"
        )

                                  
                                              
    tasks = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]

                                                                        
        if isinstance(prompt, list):
            messages = prompt
        else:
                                        
            messages = [{"role": "user", "content": str(prompt)}]

        tasks.append((idx, messages))

    results_dict = {}                               

                            
    df["accepted"] = False
    df["decision"] = None

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_one, idx, messages, client, args.model
            ): idx
            for idx, messages in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Querying model",
        ):
            idx = futures[future]
            try:
                _idx, accepted, decision = future.result()
                results_dict[_idx] = (accepted, decision)
            except Exception as e:
                print(f"[ERROR] Future for idx={idx} raised exception: {e}")
                results_dict[idx] = (False, {"status": "error", "raw": None})

            
    for idx, (accepted, decision) in results_dict.items():
        df.at[idx, "accepted"] = accepted
                             
        df.at[idx, "decision"] = json.dumps(decision, ensure_ascii=False)

    df.to_parquet(args.save_path, index=False)
    print(f"Saved results to {args.save_path}")
    df = pd.read_parquet(args.save_path)

    df = df[df['accepted'] ==True]


    ori_solver_list = df[['reward_model', 'decision', 'extra_info']].values.tolist()

    solver_list = []

    for idx, sample in tqdm(enumerate(ori_solver_list)):
        difficulty_id = sample[0]['ground_truth']
        raw = sample[1]
        text = sample[2]['text']

        solver = json.loads(raw)
        try:
            solver = json.loads(solver['raw'])
            solver['answer_type'] = sample[2]['answer_type']
            solver['difficulty_id'] = difficulty_id
            solver['text']= text
            required_keys = {
                'analysis', 'question', 'intermediate_results', 'answer',
                'solving_time_estimate', 'required_concepts', 'potential_errors',
                'difficulty_id', 'text', 'answer_type'
            }

            if set(solver.keys()) != required_keys:
                raise KeyError

            if not isinstance(solver['solving_time_estimate'], int) and not isinstance(solver['solving_time_estimate'], float):
                solver['solving_time_estimate'] = float(solver['solving_time_estimate'])


            if not isinstance(solver['question'], str) and not isinstance(solver['question'], list) and not isinstance(solver['question'], tuple):
                raise TypeError
            solver_list.append(solver)

        except:
            print('error at index:', idx)
            continue
                           
                  
                     
                                    
                                                                                                              
                                              
       
   

                
    random.shuffle(solver_list)

                              
    solver_list = sorted(
        solver_list,
        key=lambda x: x.get('difficulty_id', 0)
    )
    solver_list = [_gen_single_item(idx, sample) for idx, sample in tqdm(enumerate(solver_list))]

    answer_types = [sample['extra_info']['answer_type'] for sample in solver_list]

    stats = analyze_answer_type_balance(answer_types, batch_size=128)

    for t, s in stats.items():
        print(
            f"{t:20s} | "
            f"min={s['min']:.3f} "
            f"max={s['max']:.3f} "
            f"mean={s['mean']:.3f} "
            f"var={s['var']:.5f}"
        )

    solver_df = pd.DataFrame(solver_list)

    bad = solver_df[~solver_df["prompt"].apply(lambda v: isinstance(v, list))]
    print("bad rows:", len(bad))
    print(bad[["prompt"]].head(5))
    print(bad["prompt"].apply(type).value_counts())


                     
    def _extract_user_question(prompt_cell) -> str:
        if not isinstance(prompt_cell, list):
            return ""
        for msg in prompt_cell:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _is_categorical(extra_info_cell) -> bool:
        if not isinstance(extra_info_cell, dict):
            raise TypeError("extra_info must be a dict.")
        at = extra_info_cell.get("answer_type")
        if not isinstance(at, str) or not at.strip():
            raise ValueError("extra_info.answer_type is required and must be a non-empty string.")
        return at.strip().lower() == "categorical"

    before = len(solver_df)
    drop_mask = solver_df.apply(
        lambda row: (
            _is_categorical(row.get("extra_info"))
            and (not categorical_question_has_options(_extract_user_question(row.get("prompt"))))
        ),
        axis=1,
    )
    num_drop = int(drop_mask.sum())
    if num_drop:
        solver_df = solver_df.loc[~drop_mask].reset_index(drop=True)
    print(f"Dropped {num_drop} categorical rows without options ({num_drop / max(before, 1):.4%})")

    print('Answer Type Distribution:', Counter([sample['answer_type'] for sample in solver_df['extra_info'].tolist()]))


                                                              
    tqdm.pandas(desc="Scanning for invalid Unicode")

    bad_mask = solver_df.progress_apply(
        lambda row: any(has_surrogate_in_anything(v) for v in row.values),
        axis=1
    )

    num_bad = int(bad_mask.sum())
    print(f"Dropped {num_bad} rows ({num_bad / len(solver_df):.4%}) due to invalid Unicode")

    solver_df = solver_df.loc[~bad_mask].reset_index(drop=True)
     


    solver_df.to_parquet(args.solver_save_path, index=False)
    print(f"Saved solver results to {args.solver_save_path}, len(solver_df): {len(solver_df)}")


                                                        
     
                                                                             
     
                                                                                                               
                                        

                                                                                                        
                                        

                                                                  
                                                                                                            
