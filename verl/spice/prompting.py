from __future__ import annotations

from typing import Any, Dict, List


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

CATEGORICAL_ICL = r"""
{
  "analysis": "Create a multiple-choice question inspired by the provided document about identifying a counterfeit coin with a balance scale. Use four options A–D with exactly one correct. The answer must be the correct letter.",
  "question": "There are 23 visually identical coins. Exactly one coin is counterfeit and is either heavier or lighter than a genuine coin. You have an electronic balance scale; each weighing has exactly three possible outcomes: left side heavier, right side heavier, or balance. What is the minimum number of weighings required, in principle, to always determine which coin is counterfeit and whether it is heavier or lighter?\n\nA) 3\nB) 4\nC) 5\nD) 6",
  "intermediate_results": {
    "step1_count_cases": "There are 23 choices for the counterfeit coin and 2 possibilities (heavier or lighter), so 23 × 2 = 46 cases.",
    "step2_capacity_per_weighing": "Each weighing has 3 outcomes, so k weighings can distinguish at most 3^k outcome sequences.",
    "step3_compare": "3^3 = 27 < 46, so three weighings are not enough; 3^4 = 81 ≥ 46, so four weighings suffice in principle."
  },
  "answer": "B",
  "solving_time_estimate": 3,
  "required_concepts": [
    "Counting cases",
    "Information capacity",
    "Powers of 3"
  ],
  "potential_errors": [
    "Forgetting the heavier/lighter distinction",
    "Using 2^k instead of 3^k",
    "Concluding 3 weighings is enough without checking capacity"
  ]
}
"""

MATH_CHALLENGER_PROMPT_TEMPLATE = '''Your task is to generate a single self-contained question and its correct answer inspired by the given document.
The question must strictly satisfy the answer_type constraints.

You must output exactly one JSON object as specified below.  
All reasoning MUST be placed inside the "analysis" field of the JSON.

## Document
[BEGIN]
{text}
[END]

## Answer Type

You must generate a question whose answer has the following type:

- answer_type: {answer_type}

Rules:
- integer → JSON integer
- float → JSON number with a decimal point
- categorical → One uppercase letter corresponding to the correct option.


## Core Requirements for the Question

1. The question must be inspired by the document (but self-contained).
2. The question must not reference “the document” or “the text”.
3. The question must be understandable by someone who only sees the question.
4. The answer must be unique and consistent with the document.
5. All variables must be defined in the question itself.
6. No ambiguity.
7. Don't copy the question form the document.
8. If answer_type is categorical: write a multiple-choice question with exactly 4 options labeled A, B, C, D (one correct).
   Put each option on its own line, starting with "A) ", "B) ", "C) ", "D) ".

---

# Final Output Format (STRICT)

Your output must be **exactly one JSON object** with the following fields:

- "analysis" (string)
- "question" (string)
- "intermediate_results" (object)
- "answer" (number|string)
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

### 4. "answer" (number|string)
Type depends on answer_type:
- integer → JSON integer
- float → JSON number with a decimal point
- categorical → JSON string, one uppercase letter in {{"A","B","C","D"}}

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

def build_challenger_messages(doc_text: str, *, answer_type: str = "integer") -> List[Dict[str, Any]]:

    at = (answer_type or "integer").strip().lower()
    if at not in {"integer", "float", "categorical"}:
        at = "integer"

    example = CATEGORICAL_ICL if at == "categorical" else math_id2ICL[1]
    question = MATH_CHALLENGER_PROMPT_TEMPLATE.format(
        answer_type=at,
        text=doc_text,
        example=example,
    )
    return [
            {
                "role": "user",
                "content": question,
            }
        ]


def build_reasoner_messages(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                r"Please reason step by step, and put your final answer within \boxed{}."
            ),
        },
        {"role": "user", "content": question},
    ]
