                     
                       














from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit                                              

                                                                              
                                  
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument('--max_model_len', type=int, default=32000,
                    help='Maximum model sequence length (tokens) for the vLLM engine. Reduce to save KV cache memory.')
parser.add_argument('--n_candidates', type=int, default=int(os.getenv('VLLM_N_CANDIDATES', '10')),
                    help='Number of candidate answers to sample per question.')
args = parser.parse_args()

                                                                          
                                  
print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=args.max_model_len,
)

sample_params = vllm.SamplingParams(
    max_tokens=8192,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.n_candidates,                                                             
)

                                                                             
                                  
stop_event = threading.Event()                                       
pause_event = threading.Event()                                              

def gpu_idle_worker():




    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1)                             
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
                                                             
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

                                                                                     
                                                                               
                                                                           
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):





    return grade_answer(res1, res2)

                                                                              
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():


                                                            
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

                                     
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

                                                
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

                                           
                                               
    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] Generation completed.')

                                                                                          
    def process_single(question, golden_answer, response):

        results = [extract_boxed_content(out.text) for out in response.outputs]
                                                                              

        answer_counts = {}
        for res in results:
            if not res: continue                     
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                                                                                            
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break                                                        
                
                                                                                              
                try:
                    is_match = False
                                                       
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                                                                               
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                                                                           
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break                                         

                except Exception as e:
                                                                                       
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue                                                    
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score if majority_ans == golden_answer and score > 0.1 else 0,
            'results':  results
        }

    results_all = []
    response_idx = 0
    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except Exception as e:
                                                                               
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

                                        
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

@app.route('/related', methods=['GET'])
def related():

    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f"[server][related] Received request for task file: {name}")
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    chats = []
    items = []
    for item in data:
        text = item.get('text', '') or ''
        question = item.get('question', '') or ''
        if not text or not question:
            items.append({"text": text, "question": question, "related": False, "score": 0})
            continue
        user_content = (
            "Do you think this question `" + question + "` is related to the following text:"\
            + text + "\n\nReturn with yes or no, no other text."
        )
        chats.append([
            {"role": "system", "content": "You are a precise judge. Reply with yes or no only."},
            {"role": "user", "content": user_content},
        ])
        items.append({"text": text, "question": question})

    if chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in chats
            ]
    else:
        prompts = []

    sampling_params_related = vllm.SamplingParams(
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        stop_token_ids=[tokenizer.eos_token_id],
        n=1,
    )

    if prompts:
        responses = model.generate(prompts, sampling_params=sampling_params_related, use_tqdm=False)
    else:
        responses = []

    out_items = []
    r_idx = 0
    for base in items:
        if 'related' in base:
            out_items.append(base)
            continue
        resp = responses[r_idx]
        r_idx += 1
        text_out = resp.outputs[0].text.strip() if resp.outputs else ''
        low = text_out.lower()
        if 'yes' in low and 'no' not in low:
            related_flag = True
        elif 'no' in low and 'yes' not in low:
            related_flag = False
        else:
            related_flag = low.startswith('y') and not low.startswith('n')
        out_items.append({
            "text": base.get('text', ''),
            "question": base.get('question', ''),
            "related": bool(related_flag),
            "score": 1 if related_flag else 0,
            "raw": text_out,
        })

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(out_items, f, indent=2)

    pause_event.clear()
    print(f"[server][related] Processed {name}, results saved to {out_path}. Resuming idle worker.")
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

                                                                                     
                                  
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
                                                            
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
