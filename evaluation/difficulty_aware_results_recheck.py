import json
import os
from openai import OpenAI
from tqdm import tqdm
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str)
parser.add_argument("--model_name", type=str, default="gpt-4o")
parser.add_argument("--api_urls", type=str, default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
parser.add_argument("--api_keys", type=str, default=os.getenv("API_KEY"))
args = parser.parse_args()

api_urls = [u.strip() for u in args.api_urls.split(",") if u.strip()]
api_keys = [k.strip() for k in (args.api_keys or "").split(",") if k.strip()]
if not api_keys:
    raise ValueError("API_KEY (or --api_keys) is required.")
if len(api_urls) != len(api_keys):
    if len(api_keys) == 1:
        api_keys = api_keys * len(api_urls)
    else:
        raise ValueError("api_urls and api_keys length mismatch.")

                                      
clients = [
    OpenAI(base_url=api_url, api_key=api_key)
    for api_url, api_key in zip(api_urls, api_keys)
]


                              
                                     
                              
def call_api_with_retry(answer, response, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
                                         
            api_index = random.randint(0, len(clients) - 1)
            client = clients[api_index]

            completion = client.chat.completions.create(
                                                              
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a math answer checker."},
                    {
                        "role": "user",
                        "content": (
                            f"Hi, there is a answer: {answer}\n\n, "
                            f"and the ground truth answer is: {response}\n\n, "
                            f"please check whether the answer is correct or not, "
                            f"and return the **only** Yes or No."
                        ),
                    },
                ],
                temperature=0.1,
                timeout=20,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2        
            else:
                return "No"

    return "No"


                              
           
                              
def process_example_parallel(item):
    answer, response = item["answer"], item["response"]
    score = item["score"]
    idx = item["idx"]              

    if score >= 0.5:
        return idx, score             

    gpt_check = call_api_with_retry(answer, response)
    if "yes" in gpt_check.lower():
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
                idx, new_score = future.result()
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
