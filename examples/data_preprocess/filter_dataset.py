import argparse
import json
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd


FILTER_PROMPT_TEMPLATE = """
You are an expert dataset auditor for a difficulty-aware question-generation system.

Your task is to determine whether the given document contains enough structured, diverse, multi-span information to support generating **ALL THREE difficulty levels**:
- Easy (1–2 spans, simple inference)
- Medium (2+ spans, moderate reasoning)
- Hard (3+ distant spans, multi-step synthesis, ≥5 reasoning steps)

A document must support **all three** difficulty levels to be considered valid.  
If any difficulty level cannot be generated due to insufficient content, the document must be rejected.

---

# Document
[BEGIN]
{text}
[END]

---

# Evaluation Criteria (STRICT)

You must evaluate whether the document contains:

### 1. Sufficient spans
- **Easy:** at least 1–2 clear factual or mathematical spans
- **Medium:** at least 2 distinct spans that relate but require combination
- **Hard:** at least 3+ *distant* spans enabling multi-step global reasoning or synthesis

### 2. Diverse reasoning types
- Quantitative, logical, relational, temporal, causal, definitional, structural, or procedural information  
- Hard questions must be realistically constructible (≥5 steps)

### 3. Answerability
- Facts are explicit, not implied or missing
- No contradictions, no hallucination traps

### 4. Multi-level reasoning potential
- Text must allow increasing difficulty by combining spans in more complex ways
- Hard difficulty must be feasible WITHOUT inventing missing info

If the document only supports Easy or Easy+Medium → reject.

---

# Output Format (STRICT)

You must output exactly **one JSON object** with these fields:

- "analysis": string  
  Detailed explanation of:
  - extracted spans  
  - whether spans can support easy / medium / hard  
  - limitations or missing information  
  - reasoning about why the document passes or fails  

- "supports_easy": boolean  
- "supports_medium": boolean  
- "supports_hard": boolean  

- "final_decision": string  
  One of:
  - "accept"  (only if all three levels are supported)
  - "reject"

- "reason_for_decision": string  
  One-sentence summary of the core reason.

- "missing_requirements": array of strings  
  List any missing elements needed for full support of all difficulty levels.

---

# Acceptance Rule

A document is accepted **ONLY IF**:
- supports_easy = true  
- supports_medium = true  
- supports_hard = true  

Otherwise:
- final_decision = "reject"
"""



GENERAL_FILTER_PROMPT_TEMPLATE = """
You are an expert curator for **STEM-grade evaluation benchmarks**, comparable to
**GPQA, MMLU-Pro, and SuperGPQA**.

Your task is to audit whether the given document is a **high-quality STEM or Life-Science source**
that can support generating **ALL THREE difficulty levels** of questions
(Easy / Medium / Hard) in a **difficulty-aware question generation system**.

This filter is STRICTLY focused on the following benchmark-aligned domains:
- Mathematics
- Physics
- Chemistry
- Biology
- Engineering
- Computer Science
- Economics

Documents outside these domains, or documents that only support shallow factual,
descriptive, or narrative questions, MUST be rejected.



---

# Document
[BEGIN]
{text}
[END]
---

## Difficulty Definitions (Benchmark-Aligned)

A valid document must support **all three** levels below:

### Easy
- 1–2 **explicit** factual, mathematical, biological, or definitional spans
- Direct retrieval or single-step inference
- Typical of low-difficulty MMLU-Pro questions

### Medium
- At least 2 **distinct but related** spans
- Requires combining formulas, definitions, mechanisms, or processes
- Moderate reasoning (2–4 steps), similar to mid-level MMLU-Pro / SuperGPQA

### Hard (CRITICAL)
- At least **3 or more distant spans**
- Supports **multi-step synthesis (≥5 reasoning steps)**
- Requires chaining concepts, constraints, equations, mechanisms, or biological pathways
- Comparable in difficulty to **GPQA / GPQA-Diamond / SuperGPQA hard questions**
- Difficulty must come from **reasoning depth**, NOT linguistic complexity

⚠️ If HARD questions would require inventing missing facts, assumptions,
or external knowledge → the document FAILS.

---

## Evaluation Criteria (STRICT)

You must assess whether the document contains:

### 1. Domain Validity (STEM / Biology Gate)
- Core content is explicitly within one or more of:
  Mathematics, Physics, Chemistry, Biology, Engineering,
  Computer Science, Economics
- Biology content must be **mechanistic or quantitative**, e.g.:
  - molecular / cellular mechanisms
  - regulatory pathways
  - biochemical reactions
  - physiological systems
  - genetics or evolutionary models
- Purely descriptive biology (e.g., species lists, surface-level anatomy)
  without reasoning structure → reject
- If primarily narrative, qualitative, or non-technical → reject immediately

### 2. Sufficient and Structured Spans
- Easy: clear standalone spans (definitions, formulas, mechanisms)
- Medium: multiple related spans enabling combination or comparison
- Hard: multiple **distant**, non-redundant spans enabling global reasoning

### 3. Reasoning Diversity
The document should enable **multiple reasoning types**, such as:
- Mathematical or symbolic reasoning
- Quantitative computation or estimation
- Logical or constraint-based reasoning
- Causal or mechanistic explanation
- Structural or algorithmic reasoning
- Biological pathway or system-level reasoning
- Economic modeling or incentive-based reasoning

Hard-level questions must be realistically constructible using these.

### 4. Answerability & Scientific Rigor
- All required facts are explicitly present in the document
- No missing variables, undefined entities, or implicit assumptions
- No internal contradictions or ambiguity
- No reliance on common sense or external domain knowledge

### 5. Multi-Level Scalability
- The SAME document must allow:
  - Easy questions using isolated spans
  - Medium questions using partial integration
  - Hard questions using full-document synthesis
- If difficulty cannot be increased without hallucination → reject

---

## Output Format (STRICT)

You must output exactly **one JSON object** with the following fields:

- "analysis": string  
  A detailed, structured explanation covering:
  - Identified domain(s)
  - Extracted spans
  - Feasibility of Easy / Medium / Hard questions
  - Why HARD-level reasoning is or is not possible
  - Any critical limitations

- "supports_easy": boolean  
- "supports_medium": boolean  
- "supports_hard": boolean  

- "final_decision": string  
  Must be one of:
  - "accept"  (ONLY if all three are supported)
  - "reject"

- "reason_for_decision": string  
  One concise sentence explaining the core reason.

- "missing_requirements": array of strings  
  List any missing elements preventing full support
  (e.g., "insufficient distant spans", "no multi-step mechanistic chain").

---

## Acceptance Rule (NON-NEGOTIABLE)

A document is accepted **ONLY IF**:
- supports_easy = true
- supports_medium = true
- supports_hard = true

Otherwise:
- final_decision MUST be "reject"
"""


def load_and_concat(paths):
    dfs = []

    for path in paths:
        path = path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.lower().endswith(".jsonl") or path.lower().endswith(".json"):
            df = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        df["__source_file__"] = os.path.basename(path)           
        dfs.append(df)

    if not dfs:
        raise ValueError("No input files loaded")

    return pd.concat(dfs, ignore_index=True)


def dedup_by_semantic_similarity(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
):
    assert text_col in df.columns

    model = SentenceTransformer(model_name)

    texts = df[text_col].tolist()

                                                 
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

                              
    sim_matrix = torch.matmul(embeddings, embeddings.T)

    n = sim_matrix.size(0)
    keep = torch.ones(n, dtype=torch.bool)

    for i in range(n):
        if not keep[i]:
            continue
                                          
        dup_mask = (sim_matrix[i] >= threshold)
        dup_mask[:i+1] = False
        keep[dup_mask] = False
    torch.cuda.empty_cache()

    return df[keep.cpu().numpy()].reset_index(drop=True)


def call_filter_model(client: OpenAI, model: str, text: str, max_retries: int = 3, sleep_sec: float = 2.0):

                                                       

    prompt = GENERAL_FILTER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
                                              
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                                   
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)
            return result
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
        print(f"[{idx}] accepted={accepted} | ERROR in call_filter_model")
        return idx, accepted, decision

    supports_easy = bool(result.get("supports_easy", False))
    supports_medium = bool(result.get("supports_medium", False))
    supports_hard = bool(result.get("supports_hard", False))
    final_decision = result.get("final_decision", "").lower()

    accepted = (
        supports_easy and
        supports_medium and
        supports_hard and
        final_decision == "accept"
    )

    decision = {
        "status": "ok",
        "supports_easy": supports_easy,
        "supports_medium": supports_medium,
        "supports_hard": supports_hard,
        "final_decision": final_decision,
        "reason_for_decision": result.get("reason_for_decision", ""),
        "full_decision_text": json.dumps(result),
    }

    print(f"[{idx}] accepted={accepted} | easy={supports_easy}, medium={supports_medium}, hard={supports_hard}, decision={final_decision}")
    return idx, accepted, decision


def main():
    parser = argparse.ArgumentParser()
                                                                               
                                                                   

    parser.add_argument(
        "--parquet_path",
        type=str,
        nargs="+",
        default=None,
        help="输入 parquet 路径（支持多个；也支持逗号分隔）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getenv("OUTPUT_PATH"),
        help="过滤后输出 parquet 路径",
    )
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--batch_limit", type=int, default=300000)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    if not args.output_path:
        raise ValueError("OUTPUT_PATH (or --output_path) is required.")

            
    parquet_paths = []
    if args.parquet_path:
        for p in args.parquet_path:
            parquet_paths.extend(p.split(","))
    else:
        env_paths = os.getenv("PARQUET_PATHS")
        if not env_paths:
            raise ValueError("PARQUET_PATHS (or --parquet_path) is required.")
        parquet_paths = [p.strip() for p in env_paths.split(",") if p.strip()]

    df = load_and_concat(parquet_paths)


                                                                                                             



    if args.text_column not in df.columns:
        raise ValueError(f"text_column '{args.text_column}' not found in dataframe columns: {df.columns.tolist()}")

    if args.batch_limit is not None:
                                        
        start = 0
        end = start + args.batch_limit
        df = df.iloc[start:end]

            
    tasks = []
    for idx, row in df.iterrows():
        text = str(row[args.text_column])
        text = text[:7000]
                     
        tasks.append((idx, text))

    results_dict = {}                               

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one, idx, text, client, args.model): idx
            for idx, text in tasks
        }

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
    main()
