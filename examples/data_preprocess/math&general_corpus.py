                                                     
 
                                                                 
                                                                  
                                         
 
                                                
 
                                                                     
                                                                   
                                                                          
                                                                     
                                




import argparse
import os
import random
import re
import datasets
import pandas as pd
from datasets import Dataset

global_text_lenghs = []


id2def = {
    1: """1 — Easy:
  - Uses 1–2 spans
  - 1–2 reasoning steps
  - Simple inference or transformation""",
    2: """2 — Moderate:
  - Uses at least 2 distinct spans
  - 2–3 reasoning steps
  - Requires combining multiple facts""",
    3: """3 — Hard:
  - Uses 3+ distant spans across the document
  - 5+ reasoning steps
  - Requires global synthesis, constrained reasoning, or multi-stage calculations"""
}

math_id2ICL = {
    1: """
{
  "analysis": "Construct an easy-level AMC-style question using a single fact from the setting: there are 23 coins and exactly one is counterfeit. Only one reasoning step (subtraction) is required, satisfying the Easy difficulty definition.",
  "question": "A collection contains 23 coins. Exactly one of them is counterfeit. How many of the coins are genuine?",
  "intermediate_results": {
    "step1_count_genuine": "Since there are 23 coins in total and exactly 1 is counterfeit, the number of genuine coins is 23 − 1 = 22."
  },
  "answer": 22,
  "solving_time_estimate": 1,
  "required_concepts": [
    "Basic counting",
    "Subtraction"
  ],
  "potential_errors": [
    "Forgetting that exactly one coin is counterfeit",
    "Subtracting more than one coin"
  ]
}
""",

    2: """
{
  "analysis": "Design a moderate-level AMC-style problem requiring exactly two reasoning steps: counting the number of possible counterfeit coins and accounting for the two possible weight deviations (heavier or lighter). This combines two facts from the setting and meets the Moderate difficulty definition.",
  "question": "There are 23 visually identical coins. Exactly one coin is counterfeit and is either heavier or lighter than a genuine coin. Each possible situation is determined by choosing the counterfeit coin and whether it is heavier or lighter. How many distinct situations are possible?",
  "intermediate_results": {
    "step1_choose_coin": "There are 23 possible choices for which coin is counterfeit.",
    "step2_account_weight": "Each counterfeit coin can be either heavier or lighter, giving 23 × 2 = 46 distinct situations."
  },
  "answer": 46,
  "solving_time_estimate": 3,
  "required_concepts": [
    "Counting principles",
    "Case analysis"
  ],
  "potential_errors": [
    "Forgetting to count both heavier and lighter cases",
    "Adding instead of multiplying the cases",
    "Assuming the counterfeit coin's weight difference is known"
  ]
}
""",

    3: """
{
  "analysis": "Create a hard-level AMC-style problem requiring global synthesis and multiple reasoning steps. The problem combines: counting all counterfeit scenarios, understanding ternary outcomes of an electronic balance, and applying an information-capacity argument using powers of 3. This satisfies the Hard difficulty definition with 5+ reasoning steps.",
  "question": "There are 23 visually identical coins. Exactly one coin is counterfeit and is either heavier or lighter than a genuine coin. You have an electronic balance scale where each weighing has exactly three possible outcomes: the left side is heavier, the right side is heavier, or the two sides balance. What is the minimum number of weighings required, in principle, to always determine which coin is counterfeit and whether it is heavier or lighter?",
  "intermediate_results": {
    "step1_count_scenarios": "There are 23 choices for the counterfeit coin and 2 possible weight deviations, giving 23 × 2 = 46 scenarios.",
    "step2_outcomes_per_weighing": "Each weighing has 3 possible outcomes.",
    "step3_total_sequences": "With k weighings, the maximum number of distinct outcome sequences is 3^k.",
    "step4_test_three_weighings": "3^3 = 27, which is less than 46, so three weighings are insufficient.",
    "step5_test_four_weighings": "3^4 = 81, which is at least 46, so four weighings are sufficient in principle."
  },
  "answer": 4,
  "solving_time_estimate": 8,
  "required_concepts": [
    "Counting counterfeit scenarios",
    "Ternary outcome systems",
    "Powers of 3",
    "Information-capacity reasoning"
  ],
  "potential_errors": [
    "Forgetting to include both heavier and lighter cases",
    "Using 2^k instead of 3^k",
    "Assuming three weighings are sufficient without checking capacity",
    "Miscalculating powers of 3"
  ]
}
"""
}


general_id2ICL = {
    1: """
{
  "analysis": "Difficulty: Easy. Use one local span: once CGG repeats exceed a threshold, FMR1 is methylated and no protein is produced, leading to the fragile X phenotype. The question is a single-step conceptual inference (identify the immediate molecular consequence).",
  "question": "Fragile X (FRAX) syndrome can arise when a CGG-repeat expansion in FMR1 exceeds a threshold length. Which downstream event most directly explains why the fragile X mental retardation protein (FMRP) is not produced?\\nA. The FMR1 gene becomes methylated, preventing expression.\\nB. The CGG repeat is shortened back below the threshold during transmission.\\nC. FMRP is produced normally but rapidly degraded in the cytoplasm.\\nD. The FMR1 gene is duplicated, diluting transcription factors.",
  "intermediate_results": {
    "step1_key_mechanism": "Above-threshold CGG expansion triggers methylation of FMR1.",
    "step2_consequence": "Methylation silences expression, so no FMRP is produced."
  },
  "answer": "A",
  "solving_time_estimate": 1,
  "required_concepts": [
    "CGG repeat expansion",
    "Threshold effect",
    "DNA methylation",
    "Gene silencing",
    "Loss of FMRP"
  ],
  "potential_errors": [
    "Confusing methylation with mutation reversion",
    "Assuming the issue is protein degradation rather than gene silencing",
    "Selecting an option that changes gene copy number without support"
  ]
}
""",

    2: """
{
  "analysis": "Difficulty: Moderate. Use at least two spans: (i) the patient has a mosaic pattern with 25% premutation, 60% full mutation, 15% deletion; (ii) full mutation is associated with methylation and loss of expression (no FMRP). Construct a 2–3 step quantitative comparison: compute (full+deletion) vs premutation and reduce to a ratio.",
  "question": "A male patient carries a mosaic FMR1 pattern: 25% premutation, 60% full mutation, and 15% deletion. Assume full mutation is methylated and does not express FMR1 (no FMRP), while the premutation category is considered separately. What is the ratio of (full mutation + deletion) to premutation in this mosaic pattern?\\nA. 1:3\\nB. 3:1\\nC. 4:1\\nD. 2:1",
  "intermediate_results": {
    "step1_sum_full_and_deletion": "Full mutation + deletion = 60% + 15% = 75%.",
    "step2_form_ratio": "Ratio (75%):(25%) simplifies to 3:1."
  },
  "answer": "B",
  "solving_time_estimate": 3,
  "required_concepts": [
    "Mosaicism proportions",
    "Combining categories",
    "Ratio simplification",
    "Full mutation and methylation concept"
  ],
  "potential_errors": [
    "Forgetting to include the deletion fraction",
    "Using 60:25 instead of 75:25",
    "Failing to simplify the ratio",
    "Mixing up the order of the ratio (premutation : full+deletion)"
  ]
}
""",

    3: """
{
  "analysis": "Difficulty: Hard. Use 3+ spans and require synthesis: (1) mechanistic rule: beyond-threshold repeats → methylation → no FMRP; (2) mosaic pattern: premutation 25%, full mutation 60%, deletion 15%; (3) IQ is borderline normal (81); (4) FMRP expression in hair roots and lymphocytes is within the affected male range; (5) delay eyeblink conditioning is unusually high; (6) no correlation between FMRP expression, DNA diagnostics, and phenotype. The question requires global reconciliation of genotype–protein–phenotype discordance with 10 options (A–J).",
  "question": "A male from a classical fragile X (FRAX) family shows an atypically mild presentation: at age 28 his IQ (Raven test) is 81 (borderline normal), and his delay eyeblink conditioning yields a much higher percentage of conditioned responses than typical FRAX cohorts. Genetic testing shows an FMR1 mosaic pattern consisting of 25% premutation, 60% full mutation, and 15% deletion. In fragile X syndrome, once CGG repeats exceed a threshold length, FMR1 can become methylated and no FMRP is produced. However, FMRP expression studies in both hair roots and lymphocytes fall within the affected male range. The case report also notes no correlation between (i) FMRP expression and phenotype and (ii) DNA diagnostics and phenotype in this individual.\\n\\nWhich interpretation best reconciles ALL of these observations?\\nA. Peripheral-tissue measures (hair roots/lymphocytes) and bulk DNA mosaic percentages may not reflect the functionally relevant cell populations for cognition or learning in this individual, allowing an unusually mild phenotype despite affected-range peripheral readouts.\\nB. Full-mutation alleles must be overexpressing FMRP, which explains both low measured FMRP and high cognitive performance.\\nC. The fragile X phenotype depends only on the deletion category; therefore 15% deletion determines IQ and conditioning outcomes.\\nD. Delay eyeblink conditioning directly measures CGG repeat length, so the high conditioning rate implies no full mutation is present.\\nE. The methylation mechanism cannot occur in humans; therefore the reported explanation is incorrect.\\nF. IQ testing at age 28 is invalid for FRAX and should be ignored.\\nG. The premutation fraction guarantees normal cognition regardless of other categories.\\nH. The observed mosaic percentages mathematically force a strong positive correlation between DNA diagnostics and phenotype.\\nI. The high conditioning rate proves FMRP is high in lymphocytes, so the assay must be wrong.\\nJ. Any full mutation necessarily implies severe impairment.",
  "intermediate_results": {
    "step1_identify_discordance": "The patient shows mild cognition and strong conditioning despite affected-range peripheral FMRP and substantial full mutation/deletion.",
    "step2_use_reported_correlations": "The report explicitly states no correlation between FMRP expression, DNA diagnostics, and phenotype in this individual.",
    "step3_eliminate_contradictions": "Options B–J impose deterministic or contradictory mechanisms inconsistent with the described observations.",
    "step4_select_integrative_inference": "Option A coherently explains how peripheral measures and bulk DNA mosaicism may fail to predict cognition in this exceptional case."
  },
  "answer": "A",
  "solving_time_estimate": 10,
  "required_concepts": [
    "Fragile X syndrome mechanism",
    "CGG repeat expansion and methylation",
    "FMR1 mosaicism",
    "Genotype–phenotype discordance",
    "Peripheral vs neural relevance",
    "Delay eyeblink conditioning"
  ],
  "potential_errors": [
    "Assuming genotype percentages deterministically predict cognition",
    "Equating peripheral FMRP with brain function",
    "Ignoring the explicitly stated lack of correlation",
    "Treating behavioral assays as direct genetic measurements",
    "Over-interpreting mosaic ratios without functional context"
  ]
}
"""
}





MATH_CHALLENGER_PROMPT_TEMPLATE = '''Your task is to generate a single self-contained question and its correct answer inspired by the given document.
The question must strictly satisfy both the difficulty level and the answer_type constraints.

You must output exactly one JSON object as specified below.  
All reasoning MUST be placed inside the "analysis" field of the JSON.

## Document
[BEGIN]
{text}
[END]

## Difficulty Level

You are given a target difficulty level:

- difficulty_id: {difficulty_id}

You must follow these operational definitions:

{definition}

Your generated question and solution process must match the target difficulty level as closely as possible.

## Answer Type

You must generate a question whose answer has the following type:

- answer_type: {answer_type}

Rules:
- integer → JSON integer
- float → JSON number with a decimal point

## Core Requirements for the Question

1. The question must be inspired by the document (but self-contained).
2. The question must not reference “the document” or “the text”.
3. The question must be understandable by someone who only sees the question.
4. The reasoning steps must match the difficulty level.
5. The question must combine the number of spans required by difficulty level.
6. The answer must be unique and consistent with the document.
7. All variables must be defined in the question itself.
8. No ambiguity.
9. Don't copy the question form the document.

---

# Final Output Format (STRICT)

Your output must be **exactly one JSON object** with the following fields:

- "analysis" (string)
- "question" (string)
- "intermediate_results" (object)
- "answer" (number)
- "solving_time_estimate" (number)
- "required_concepts" (array of strings)
- "potential_errors" (array of strings)

## Field Specifications

### 1. "analysis" (string)
This field contains your full internal reasoning:
- selection of spans
- mapping to difficulty
- design of question
- intermediate calculations
- validation of uniqueness
- check for answer_type

This is the ONLY place where long reasoning is allowed.

### 2. "question" (string)
- A single natural-language exam-style question.
- Completely self-contained.
- No reasoning, no hints, no metadata.

### 3. "intermediate_results" (object)
- Keys: short step names
- Values: 1–20 sentence summaries of key reasoning steps

### 4. "answer" (number)
- Must be numeric
- Must satisfy the answer_type

### 5. "solving_time_estimate" (number)
Minutes required.

### 6. "required_concepts" (array of strings)
1–10 relevant concepts from document.

### 7. "potential_errors" (array of strings)
1–10 potential student mistakes.

---

# Example (do not copy; only follow structure; the content and answer type (Integer vs Float) in the example below are for structure demonstration only. The example is based on a generic Physics document. Your output must be based on the provided [Document] above and strictly follow the "difficulty_id" and "answer_type" assigned to you.)

## Example Input
[BEGIN]
# The Most Difficult Problem in South Vietnam 2010\\n\\n## Problem Statement\\n\\nYou have a total of 23 coins, among which one is a counterfeit coin. The counterfeit coin is either lighter or heavier than the real coins. You have access to an electronic balance scale that can be used three times. The scale shows whether the left side is heavier, the right side is heavier, or if both sides are equal. Your task is to find the counterfeit coin.\\n\\n### Discussion\\n\\n#### giacat (2010-09-08 04:46:20)\\n\\n**Member**\\n- Registered: 2010-09-02\\n- Posts: 2\\n\\nThe problem involves finding the counterfeit coin among 23 coins using an electronic balance scale three times. The scale provides a deviation value: positive if the left side is heavier, negative if the right side is heavier, and zero if both sides are equal.\\n\\n#### bob bundy (2010-09-08 07:51:10)\\n\\n**Moderator**\\n- Registered: 2010-06-20\\n- Posts: 7,736\\n\\n**Re: The Most Difficult Problem in South Vietnam 2010**\\n\\nHi giacat,\\n\\nThis problem is quite challenging. If the balance scale is like the one described, it might be possible to solve the problem with fewer coins, such as 12. It would be helpful to know the weight of a real coin.\\n\\nBob\\n\\n*Children are not defined by school... The Fonz*\\n\\n*You cannot teach a man anything; you can only help him find it within himself... Galileo Galilei*\\n\\n#### soroban (2010-09-08 18:10:47)\\n\\n**Member**\\n- Registered: 2007-03-09\\n- Posts: 452\\n\\n**Re: The Most Difficult Problem in South Vietnam 2010**\\n\\nGiacat\'s images were submitted for reference.\\n\\n### Solution Approach\\n\\nTo solve this problem, we can use a strategy that involves dividing the coins into groups and using the balance scale to systematically eliminate possibilities. Here is a step-by-step approach:\\n\\n1. **First Weighing**: Divide the 23 coins into three groups: 8 coins, 8 coins, and 7 coins. Weigh the first group against the second group.\\n - If they balance, the counterfeit coin is in the group of 7 coins.\\n - If they do not balance, the counterfeit coin is in the heavier or lighter group, depending on whether the counterfeit is heavier or lighter.\\n\\n2. **Second Weighing**: Take the group that contains the counterfeit coin (either 8 or 7 coins) and divide it into three smaller groups. For example, if you have 8 coins, divide them into groups of 3, 3, and 2. If you have 7 coins, divide them into groups of 3, 3, and 1.\\n - Weigh two of the smaller groups against each other.\\n - If they balance, the counterfeit coin is in the remaining group.\\n - If they do not balance, the counterfeit coin is in the heavier or lighter group.\\n\\n3. **Third Weighing**: Take the remaining group that contains the counterfeit coin (either 3 or 2 coins) and weigh two of the coins against each other.\\n - If they balance, the counterfeit coin is the one not weighed.\\n - If they do not balance, the counterfeit coin is the heavier or lighter one, depending on the previous weighings.\\n\\nBy following this method, you can identify the counterfeit coin within three weighings.\\n\\n### Conclusion\\n\\nThis problem requires careful planning and logical deduction to identify the counterfeit coin using the balance scale effectively. The key is to reduce the number of possibilities systematically with each weighing.
[END]

## Example Output

{example}
'''




GENERAL_CHALLENGER_PROMPT_TEMPLATE = '''Your task is to generate a single self-contained question and its correct answer inspired by the given document.
The question must strictly satisfy both the difficulty level and the answer_type constraints.

You must output exactly one JSON object as specified below.  
All reasoning MUST be placed inside the "analysis" field of the JSON.

## Document
[BEGIN]
{text}
[END]

## Difficulty Level

You are given a target difficulty level:

- difficulty_id: {difficulty_id}

You must follow these operational definitions:

{definition}

Your generated question and solution process must match the target difficulty level as closely as possible.

## Answer Type

You must generate a question whose answer has the following type:

- answer_type: {answer_type}

Rules:
- categorical → One uppercase letter corresponding to the correct option.


## Core Requirements for the Question

1. The question must be inspired by the document (but self-contained).
2. The question must not reference “the document” or “the text”.
3. The question must be understandable by someone who only sees the question.
4. The reasoning steps must match the difficulty level.
5. The question must combine the number of spans required by difficulty level.
6. The answer must be unique and consistent with the document.
7. All variables must be defined in the question itself.
8. No ambiguity.
9. Don't copy the question form the document.

---

# Final Output Format (STRICT)

Your output must be **exactly one JSON object** with the following fields:

- "analysis" (string)
- "question" (string)
- "intermediate_results" (object)
- "answer" (string)
- "solving_time_estimate" (number)
- "required_concepts" (array of strings)
- "potential_errors" (array of strings)

## Field Specifications

### 1. "analysis" (string)
This field contains your full internal reasoning:
- selection of spans
- mapping to difficulty
- design of question
- intermediate calculations
- validation of uniqueness
- check for answer_type

This is the ONLY place where long reasoning is allowed.

### 2. "question" (string)
- A single natural-language exam-style question.
- Completely self-contained.
- No reasoning, no hints, no metadata.

### 3. "intermediate_results" (object)
- Keys: short step names
- Values: 1–20 sentence summaries of key reasoning steps

### 4. "answer" (categorical)
- Must be a single uppercase letter

### 5. "solving_time_estimate" (number)
Minutes required.

### 6. "required_concepts" (array of strings)
1–10 relevant concepts from document.

### 7. "potential_errors" (array of strings)
1–10 potential student mistakes.

---

# Example (do not copy; only follow structure; the content and answer type (Integer vs Float) in the example below are for structure demonstration only. The example is based on a generic Physics document. Your output must be based on the provided [Document] above and strictly follow the "difficulty_id" and "answer_type" assigned to you.)

## Example Input
[BEGIN]
Fragile X (FRAX) syndrome is a commonly inherited form of mental retardation resulting from the lack of expression of the fragile X mental retardation protein (FMRP). It is caused by a stretch of CGG repeats within the fragile X gene, which can be unstable in length as it is transmitted from generation to generation. Once the repeat exceeds a threshold length, the FMR1 gene is methylated and no protein is produced resulting in the fragile X phenotype. The consequences of FMRP absence in the mechanisms underlying mental retardation are unknown. We have identified a male patient in a classical FRAX family without the characteristic FRAX phenotype. His intelligence quotient (IQ) is borderline normal despite the presence of a mosaic pattern of a pre-mutation (25%), full mutation (60%) and a deletion (15%) in the FMR1 gene. The cognitive performance was determined at the age of 28 by the Raven test and his IQ was 81. However, FMRP expression studies in both hair roots and lymphocytes, determined at the same time as the IQ test, were within the affected male range. The percentage of conditioned responses after delay eyeblink conditioning was much higher than the average percentage measured in FRAX studies. Moreover, this patient showed no correlation between FMRP expression and phenotype and no correlation between DNA diagnostics and phenotype. Additional Metadata Keywords Delay eyeblink conditioning, Exceptional phenotype, Fragile X syndrome Persistent URL, Journal Clinical Genetics: an international journal of genetics and molecular medicine Govaerts, L.C, Smit, A.E, Saris, J.J, VanderWerf, F, Willemsen, R, Bakker, C.E, … Oostra, B.A. (2007). Exceptional good cognitive and phenotypic profile in a male carrying a mosaic mutation in the FMR1 gene. Clinical Genetics: an international journal of genetics and molecular medicine, 72(2), 138–144. doi:10.1111/j.1399-0004.2007.00829.x
[END]

## Example Output

{example}
'''

def _gen_single_item(text: str, idx: int, difficulty_id: int, answer_type: str, _type: str):

                                                           
    id2level = {1: 'Easy', 2: 'Moderate', 3: 'Hard'}

    if _type == 'math':
        question = MATH_CHALLENGER_PROMPT_TEMPLATE.format(
            difficulty_id=id2level[difficulty_id],
            answer_type=answer_type,
            text=text,
            definition=id2def[difficulty_id],
            example=math_id2ICL[difficulty_id],
        )
    else:
        question = GENERAL_CHALLENGER_PROMPT_TEMPLATE.format(
            difficulty_id=id2level[difficulty_id],
            answer_type=answer_type,
            text=text,
            definition=id2def[difficulty_id],
            example=general_id2ICL[difficulty_id],
        )

                     
    return {
        "data_source": "questioner_given_difficulty_id",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": _type,
        "reward_model": {"style": "rule", "ground_truth": difficulty_id},
        "extra_info": {
            "split": "train",
            "index": idx,                    
            "answer_type": answer_type,
            "text": text,
        },
    }


def process_batch_fn(batch, indices, difficulties=(1, 2, 3), k_per_doc=None, int_ratio=0.8):







    out = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    n = len(batch["text"])
    for i in range(n):
        global_text_lenghs.append(len(batch["text"][i]))
        text = (batch["text"][i] or "")[:30000]
        doc_idx = indices[i] if isinstance(indices, list) else indices[i]

                       
        if k_per_doc is not None and k_per_doc < len(difficulties):
            diffs = random.sample(list(difficulties), k_per_doc)
        else:
            diffs = list(difficulties)

        for d in sorted(diffs):
            if batch['domain'][i] == 'math':
                type = 'math'
                answer_type = "integer" if random.random() < int_ratio else "float"

            else:
                type = 'general'
                answer_type = 'categorical'
            item = _gen_single_item(text=text, idx=doc_idx, difficulty_id=d, answer_type=answer_type, _type=type)
            for k in out.keys():
                out[k].append(item[k])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--math_dataset_path",
        default=os.getenv("MATH_DATASET_PATH"),
        help="The local path to the raw dataset, if it exists.",
    )

    parser.add_argument(
        "--general_dataset_path",
        default=os.getenv("GENERAL_DATASET_PATH"),
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=os.getenv("LOCAL_SAVE_DIR"),
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--difficulties",
        default="1,2,3",
        help="Comma-separated difficulty ids to expand per document (default: all 1..3).",
    )
    parser.add_argument(
        "--k_per_doc",
        type=int,
        default=None,
        help="If set, randomly sample k difficulties per document instead of all to control dataset size.",
    )
    parser.add_argument(
        "--int_ratio",
        type=float,
        default=0.8,
        help="Probability of integer answer type; remainder uses float.",
    )

    args = parser.parse_args()
    if not args.math_dataset_path or not args.general_dataset_path or not args.local_save_dir:
        raise ValueError("MATH_DATASET_PATH, GENERAL_DATASET_PATH, and LOCAL_SAVE_DIR are required.")

                                           
    math_df = pd.read_parquet(args.math_dataset_path).head(10000)

    math_df['domain'] = 'math'
    general_df = pd.read_parquet(args.general_dataset_path).head(10000)
    general_df['domain'] = 'general'
     
    df = pd.concat([math_df, general_df], ignore_index=True)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)

                                
    diffs = tuple(int(x) for x in re.split(r"[，,\s]+", args.difficulties.strip()) if x)
    train_dataset = dataset.map(
        lambda batch, idxs: process_batch_fn(batch, idxs, difficulties=diffs, k_per_doc=args.k_per_doc, int_ratio=args.int_ratio),
        with_indices=True,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print('da')
                    
    os.makedirs(args.local_save_dir, exist_ok=True)
    save_path = os.path.join(args.local_save_dir, "train.parquet")
    train_dataset.to_parquet(save_path)
    print(f"Saved processed dataset to: {save_path}")
    print(f"Dataset size: {len(train_dataset)}")
