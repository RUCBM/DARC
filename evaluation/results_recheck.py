import json
import argparse
import os
from tqdm import tqdm

if __package__:
    from .recheck_common import JudgeError, create_clients, judge_answer
else:
    from recheck_common import JudgeError, create_clients, judge_answer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--api_urls", type=str, default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
parser.add_argument("--api_keys", type=str, default=os.getenv("API_KEY"))
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")
clients = create_clients(args.api_urls, args.api_keys)


def process_example(answer, response):
    return judge_answer(clients, answer, response)


new_results = []
for model_name in [args.model_name]:
    for dataset in [
    "math",
    "gsm8k", 
    "amc",
    "minerva",
    "olympiad",
    "aime2024",
    "aime2025",
    ]:
        with open(f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}.json', 'r') as f:
            results = json.load(f)

        for i in tqdm(range(len(results)-1)):
            if results[i]['score'] < 0.5:
                try:
                    is_correct = process_example(results[i]['answer'], results[i]['response'])
                except JudgeError as exc:
                    raise JudgeError(f"{dataset}, sample {i}: {exc}") from None
                if is_correct:
                    results[i]['score'] = 1
        new_results.append({
            'model': model_name,
            'dataset': dataset,
            'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
        })
        print(new_results)
        with open(f'final_results.jsonl', 'a') as f:
            json.dump({
                'model': model_name,
                'dataset': dataset,
                'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
            }, f)
            f.write('\n')
