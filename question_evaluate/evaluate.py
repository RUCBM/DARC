#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    This script evaluates generated answers against golden answers for a set of questions.
    It uses vLLM for efficient generation and a robust, timed grading mechanism to score the results.
    The script is designed to run as a batch job, often in parallel across multiple GPUs.

Refactoring Notes:
    - Replaced 'timeout-decorator' with the thread-safe 'stopit' library to provide robust
      timeout protection for the grading function without causing errors.
    - Optimized the answer comparison logic to perform cheap checks first, only calling the
      expensive grading function when necessary.
    - Improved error handling and code structure for better readability and stability.

Setup:
    pip install stopit transformers torch vllm

Example Usage (in a shell script):
    # This would run the script for GPU 0, with a specific model and save name.
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "Qwen/Qwen3-4B-Base" --suffix 0 --save_name "my_experiment" &
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os
import stopit                                                           
from mathruler.grader import extract_boxed_content, grade_answer

                          
parser = argparse.ArgumentParser(description="Evaluate generated questions using vLLM.")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base", help="Path to the model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=9, help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
args = parser.parse_args()

                             
STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    raise ValueError("STORAGE_PATH is required.")
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"

                                            
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):




                                                                            
    return grade_answer(res1, res2)

                           

                          
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
                                                                     
    os.remove(INPUT_FILE)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
    exit()

                                                 
correct_data = [item for item in data if item.get('score') == 0]
if not correct_data:
    print(f"[{args.suffix}] No new questions to process (score=0). Exiting.")
                                                       
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

questions = [item["question"] for item in correct_data]
answers = [item["answer"] for item in correct_data]
print(f"[{args.suffix}] Found {len(questions)} questions to process.")

                                   
print(f"[{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    seed=int(args.suffix),
)
sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

                       
print(f"[{args.suffix}] Generating {args.num_samples} samples for each question...")
chats = [[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": q}] for q in questions]

if tokenizer.chat_template:
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True) for chat in chats]
else:
    prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] for chat in chats]

responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
print(f"[{args.suffix}] Generation complete.")

                                
results_all = []
print(f"[{args.suffix}] Grading responses...")
for response, golden_answer, question in zip(responses, answers, questions):
    try:
                                                              
        results = [extract_boxed_content(output.text) for output in response.outputs]
        results = [res for res in results if res]                                

        if not results:
            print(f"[{args.suffix}] WARNING: No valid boxed answers found for question: '{question[:50]}...'")
            continue

        answer_counts = {}
        for result in results:
            matched = False
            for existing_answer in answer_counts:
                                                                       
                if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                    answer_counts[existing_answer] += 1
                    matched = True
                    break
                
                                                                        
                                                            
                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{result[:30]}...' vs '{existing_answer[:30]}...'")
                    continue                                   
                
                if match_1:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{existing_answer[:30]}...' vs '{result[:30]}...'")
                    continue

                if match_2:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

            if not matched:
                answer_counts[result] = 1

        if not answer_counts:
            continue

                                                     
        majority_answer = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_answer]
        score = max_count / len(results)

                                                                          
        if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
            continue

        results_all.append({
            "question": question,
            "answer": majority_answer,
            "score": score,
            'results': results
        })

    except Exception as e:
        print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
        continue

                       
print(f"[{args.suffix}] Processed {len(results_all)} questions. Saving results to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{args.suffix}] Script finished.")
