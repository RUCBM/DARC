import datasets
import json
import re
import random
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


valid_num = 0
max_num = 0
def _model_suffix(model_path: str, n: int = 20) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path))
    return safe[-n:] if len(safe) > n else safe

def extract_last_boxed(text):
    global max_num, valid_num
    max_num += 1
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        valid_num += 1
        return matches[-1].group(1)
    return None

def extract_last_final_answer(text):
    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = r'The answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None

def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

def get_prediction(output):
    solution = extract_solution(output)
    if solution is None:
        return random.choice(['A', 'B', 'C', 'D'])
    for option in ['A', 'B', 'C', 'D']:
        if option in solution:
            return option
    return random.choice(['A', 'B', 'C', 'D'])

if __name__ == "__main__":
                 
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_file", type=str, default=None, help="File to save results")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = f"outputs_gpqa_diamond_{_model_suffix(args.model_path)}.json"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
        disable_cascade_attn=True,
    )

                            

    print('start loading dataset')
    dataset = datasets.load_dataset('fingertap/GPQA-Diamond')
    success, fail = 0, 0
    answers = []
    
    print('----------------- Start Answering -------------------')
    
    entries = [entry for entry in dataset['test']]
    prompts = []
    for entry in entries:
        query = entry['question']
        messages = [{
            "role": "user",
            "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
        }]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "user: " + query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=20000)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, entry, output in zip(prompts, entries, outputs):
        answer = output.outputs[0].text

        print(f"prompt: {prompt} answer: {answer} ")

        entry['solution'] = answer
        answers.append(entry)

        prediction = get_prediction(answer)
        if entry["answer"] == prediction:
            success += 1
        else:
            fail += 1


    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    with open("final_results.jsonl", "a", encoding="utf-8") as f:
        json.dump(
            {"dataset": "gpqa_diamond", "model": args.model_path, "accuracy": round(success / (success + fail) * 100, 2)},
            f,
            ensure_ascii=False,
        )
        f.write("\n")
    print("Overall Accuracy:", success / (success + fail))
    print("valid fraction:", valid_num / max_num)