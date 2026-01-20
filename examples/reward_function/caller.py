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

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i):
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    datas = split_list(data,4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(4)]

        for future in as_completed(futures):
            print(future.result())

    for future in as_completed(futures):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))

    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    scores = [{"overall": min(item["score"],1-item["score"]) if item['question'] else -1,"format": 1 if item['question'] else 0,"accuracy": 1 if item['answer'] else 0} for item in final_results]
    return scores








