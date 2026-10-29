import json
import argparse
import os
from tqdm import tqdm

if __package__:
    from .recheck_common import JudgeError, create_clients, judge_answer
else:
    from recheck_common import JudgeError, create_clients, judge_answer

from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str)
parser.add_argument("--model_name", type=str, default="gpt-4o")
parser.add_argument("--api_urls", type=str, default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
parser.add_argument("--api_keys", type=str, default=os.getenv("API_KEY"))
args = parser.parse_args()

clients = create_clients(args.api_urls, args.api_keys)


def call_api_with_retry(answer, response, max_retries=5):
    return judge_answer(clients, answer, response, max_attempts=max_retries)


def process_example_parallel(item):
    answer, response = item["answer"], item["response"]
    score = item["score"]
    idx = item["idx"]              

    if score >= 0.5:
        return idx, score             

    try:
        is_correct = call_api_with_retry(answer, response)
    except JudgeError as exc:
        raise JudgeError(f"sample {idx}: {exc}") from None
    if is_correct:
        score = 1

    return idx, score


                              
             
                              
new_results = []

for model_name in [args.model_name]:
    for dataset in ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]:

        with open(f'{args.result_path}/results_{dataset}.json', 'r') as f:
            results = json.load(f)

                                         
        items = [
            {
                "idx": i,
                "answer": results[i]['answer'],
                "response": results[i]['response'],
                "score": results[i]['score']
            }
            for i in range(len(results) - 1)
        ]

              
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_example_parallel, item) for item in items]

            for future in tqdm(as_completed(futures), total=len(items), desc=f"Processing {dataset}"):
                try:
                    idx, new_score = future.result()
                except JudgeError as exc:
                    for pending in futures:
                        pending.cancel()
                    raise JudgeError(f"{dataset}, {exc}") from None
                results[idx]['score'] = new_score

        avg_score = round(sum(r['score'] for r in results[:-1]) / len(results[:-1]) * 100, 2)

        new_results.append({
            "model": model_name,
            "dataset": dataset,
            "score": avg_score
        })

        print(new_results)

        with open(f'{args.result_path}/final_results.json', "a") as f:
            json.dump({"model": model_name, "dataset": dataset, "score": avg_score}, f)
            f.write("\n")
