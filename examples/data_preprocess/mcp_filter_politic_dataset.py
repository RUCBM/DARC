import argparse
import json
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


FILTER_PROMPT_TEMPLATE = """你是一个内容安全审核助手。任务：判断给定文本是否涉及**中国政治/涉政**内容。

判定为涉政（is_political=true）的典型情况包括但不限于：
- 中国党政机关、领导人、政治制度、政策方针、意识形态、重大政治事件
- 政治敏感议题/运动/示威/维权/选举/政权更迭/分裂与统一相关讨论
- 对中国政治人物/党政机构的评价、攻击、讽刺、号召、动员
- 传播政治谣言/煽动性政治内容/组织动员信息

不涉政（is_political=false）的情况包括但不限于：
- 纯技术/学术/生活/商业内容，未讨论中国政治议题
- 提到“中国”“北京”等地名但语境与政治无关（如旅游、地址、物流）
- 历史/文化内容若不涉及当代中国政治议题，也可判为不涉政

输出要求（非常重要）：
- 只输出严格合法的 JSON（不要代码块，不要额外文字）
- JSON schema 固定为：
  {{"is_political": boolean, "reason": string}}

待判断文本如下：
{text}
"""


def _extract_json_object(s: str):

    s = (s or "").strip()

                           
    if s.startswith("```"):
                
        s = s.strip().strip("`").strip()
                   
        if s.lower().startswith("json"):
            s = s[4:].strip()

            
    try:
        return json.loads(s)
    except Exception:
        pass

                       
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = s[l:r + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found.")


def call_filter_model(client: OpenAI, model: str, text: str, max_retries: int = 3, sleep_sec: float = 2.0):

    prompt = FILTER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            result = _extract_json_object(content)

                   
            is_political = bool(result.get("is_political", False))
            reason = str(result.get("reason", "")).strip()

            return {"is_political": is_political, "reason": reason}
        except Exception as e:
            print(f"[WARN] call_filter_model error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(sleep_sec)


def process_one(idx, text, client, model):

    result = call_filter_model(client, model, text)

    if result is None:
                                 
        accepted = False
        decision = {"status": "error", "raw": None}
        print(f"[{idx}] kept={accepted} | ERROR in call_filter_model")
        return idx, accepted, decision

    is_political = bool(result.get("is_political", False))
    reason = result.get("reason", "")

               
    accepted = (not is_political)

    decision = {
        "status": "ok",
        "is_political": is_political,
        "reason": reason,
        "full_decision_text": json.dumps(result, ensure_ascii=False),
    }

    print(f"[{idx}] kept={accepted} | is_political={is_political}")
    return idx, accepted, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default="./unstable_all.jsonl", help="输入 parquet 路径")
    parser.add_argument("--output_path", type=str, default="./unstable_all_filter.parquet", help="过滤后输出 parquet 路径")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY"), help="API key")
    parser.add_argument("--base_url", type=str, default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"), help="API base URL，例如 https://api.openai.com/v1 或你的自定义网关")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="用于过滤的模型名称")
    parser.add_argument("--text_column", type=str, default="text", help="存放文档内容的列名")
    parser.add_argument("--num_workers", type=int, default=32, help="并发线程数")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API_KEY (or --api_key) is required.")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    if 'parquet' in args.parquet_path.lower():
        df = pd.read_parquet(args.parquet_path)
    else:
        df = pd.read_json(args.parquet_path, lines=True)
    if args.text_column not in df.columns:
        df[args.text_column] = df['question'] + df['answer']

    tasks = []
    for idx, row in df.iterrows():
        text = str(row[args.text_column])
        tasks.append((idx, text))

    results_dict = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, idx, text, client, args.model): idx for idx, text in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering docs"):
            idx = futures[future]
            try:
                _idx, accepted, decision = future.result()
                results_dict[_idx] = (accepted, decision)
            except Exception as e:
                print(f"[ERROR] Future for idx={idx} raised exception: {e}")
                results_dict[idx] = (False, {"status": "error", "raw": None})

    keep_mask = []
    decisions = []
    for idx in df.index:
        accepted, decision = results_dict.get(idx, (False, {"status": "error", "raw": None}))
        keep_mask.append(accepted)
        decisions.append(decision)

    df["filter_meta"] = decisions
    df_filtered = df[pd.Series(keep_mask, index=df.index)]

    print(f"Original rows: {len(df)}, kept: {len(df_filtered)}")
    df_filtered.to_parquet(args.output_path, index=False)
    print(f"Saved filtered parquet to: {args.output_path}")


if __name__ == "__main__":
    import pandas as pd

                
    df = pd.read_parquet("unstable_all_filter.parquet")
    df = df[['id', 'question', 'answer',]]
    print('da')
                         
    df.to_json(
        "unstable_all_filter.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    print("Saved to unstable_all_filter.jsonl")

            
