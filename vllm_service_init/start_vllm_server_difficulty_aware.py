                     
                       














from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import re
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit                                              

                                                                              
                                  
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.85,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument('--max_model_len', type=int, default=8192,
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

                                                                                 
solver_temp = float(os.getenv('SOLVER_TEMPERATURE', '1.0'))
solver_top_p = float(os.getenv('SOLVER_TOP_P', '1.0'))
solver_top_k = int(os.getenv('SOLVER_TOP_K', '40'))
solver_max_tokens = int(os.getenv('SOLVER_MAX_TOKENS', '4096'))

sample_params = vllm.SamplingParams(
    max_tokens=solver_max_tokens,
    temperature=solver_temp,
    top_p=solver_top_p,
    top_k=solver_top_k,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.n_candidates,                                                             
)

                                             
LOG_TRUNCATE = os.getenv("LOG_TRUNCATE", "1") == "1"

def _trunc(obj, n: int = 120) -> str:
    try:
        s = str(obj)
    except Exception:
        s = repr(obj)
    if not LOG_TRUNCATE:
        return s
    return s if len(s) <= n else s[:n] + '...'

def _top_counts(d: dict, k: int = 3):
    try:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
        return [( _trunc(ans, 60), cnt) for ans, cnt in items]
    except Exception:
        return []

CATEGORICAL_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer option within \\boxed{}."
    " Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
)

def _is_invalid_candidate(ans: object) -> bool:

    if not isinstance(ans, str):
        return True
    s = ans.strip()
    if not s:
        return True
    return s.lower() == "none"


def _chat_to_prompt_fallback(chat: list[dict]) -> str:

    lines = []
    for msg in chat:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _looks_like_categorical_question(question: str) -> bool:

    if not isinstance(question, str) or not question.strip():
        return False
                                                                                     
    hits = re.findall(r"(?m)^\s*([A-J])[\.\)]\s+", question.upper())
    return len(set(hits)) >= 3


def _is_categorical_item(item: dict, question: str, golden_answer: str | None = None) -> bool:
    at = (item.get("answer_type") or "").strip().lower() if isinstance(item, dict) else ""
    if at == "categorical":
        return True
    if isinstance(golden_answer, str) and re.fullmatch(r"[A-J]", golden_answer.strip().upper() or ""):
        return True
    return _looks_like_categorical_question(question)

                                                                             
                                  
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
    for i, (q, a, item) in enumerate(zip(questions, answers, data)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            if _is_categorical_item(item, q, golden_answer=a):
                query = q + CATEGORICAL_INSTRUCTION
                valid_chats.append([{'role': 'user', 'content': query}])
            else:
                valid_chats.append([
                    {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                    {'role': 'user',   'content': q}
                ])
    print('[server] Valid chat prompts have been prepared.')
    print('[server] Gen config:',
          f"n_candidates={sample_params.n}",
          f"temp={sample_params.temperature}",
          f"top_p={sample_params.top_p}",
          f"top_k={sample_params.top_k}",
          f"max_tokens={sample_params.max_tokens}")

                                           
                                               
    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [_chat_to_prompt_fallback(chat) for chat in valid_chats]
        t0 = time.time()
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
        dt = time.time() - t0
        print(f"[server] Generation took {dt:.2f}s for {len(prompts)} prompts.")
    else:
        responses = []
    print('[server] Generation completed.')

                                                                                              
    def process_single(question, golden_answer, response):






                                                           
                                                                                               
                                                                                                
        raw_results = [extract_boxed_content(out.text) for out in response.outputs]
        results = [r for r in raw_results if r]                                         
        has_any_valid = any(not _is_invalid_candidate(r) for r in results)
        if not has_any_valid:
            print('  [post] question=', _trunc(question, 300))
            print('  [post] golden =', _trunc(golden_answer, 100))
            print('  [post] results= 0 cands; top counts= []')
            print('  [post] majority= (none)')
            print('  [post] success =', "0/0", "rate=-1.000")
            return {
                'question': question,
                'answer':   '',
                'majority_fraction': 0.0,
                'score':    -1.0,
                'results':  [],
            }

                                                                
        answer_counts = {}
        for r in results:
            if not r:
                continue
            matched_to_existing = False
            for exist in list(answer_counts.keys()):
                                                           
                if r == exist or ('no ' in r.lower() and 'no ' in exist.lower()):
                    answer_counts[exist] += 1
                    matched_to_existing = True
                    break
                try:
                    is_match = False
                    m1 = grade_answer_with_timeout(r, exist, timeout=10)
                    if m1 == 'TIMED_OUT':
                        print(f"      [cluster] TIMEOUT '{r[:30]}...' vs '{exist[:30]}...'")
                    elif m1:
                        is_match = True
                    if not is_match:
                        m2 = grade_answer_with_timeout(exist, r, timeout=10)
                        if m2 == 'TIMED_OUT':
                            print(f"      [cluster] TIMEOUT '{exist[:30]}...' vs '{r[:30]}...'")
                        elif m2:
                            is_match = True
                    if is_match:
                        answer_counts[exist] += 1
                        matched_to_existing = True
                        break
                except Exception as e:
                    print(f"      [cluster] ERROR comparing '{r[:30]}...' and '{exist[:30]}...': {e}")
            if not matched_to_existing:
                answer_counts[r] = 1

        if answer_counts:
            majority_ans = max(answer_counts, key=answer_counts.get)
            majority_count = answer_counts[majority_ans]
        else:
            majority_ans = ''
            majority_count = 0

                                                                                     
        matches = 0
        total = len(results)
        for r in results:
            if not r:
                continue
            if _is_invalid_candidate(r):
                                                                        
                continue
            matched = False
                                  
            if r == golden_answer or ('no ' in r.lower() and 'no ' in golden_answer.lower()):
                matched = True
            if not matched:
                try:
                    m1 = grade_answer_with_timeout(r, golden_answer, timeout=10)
                    if m1 == 'TIMED_OUT':
                                                                       
                        m2 = grade_answer_with_timeout(golden_answer, r, timeout=10)
                        matched = (m2 != 'TIMED_OUT') and bool(m2)
                    else:
                        matched = bool(m1)
                    if not matched:
                                                                            
                        m2 = grade_answer_with_timeout(golden_answer, r, timeout=10)
                        matched = (m2 != 'TIMED_OUT') and bool(m2)
                except Exception as e:
                    print(f"      [grader] ERROR comparing to golden: '{r[:30]}...' vs '{golden_answer[:30]}...': {e}")
            if matched:
                matches += 1

        success_rate = (matches / total) if total > 0 else 0.0
        majority_fraction = (majority_count / total) if total > 0 else 0.0

                              
        print('  [post] question=', _trunc(question, 300))
        print('  [post] golden =', _trunc(golden_answer, 100))
        print('  [post] results=', len(results), 'cands; top counts=', _top_counts(answer_counts))
        print('  [post] majority=', _trunc(majority_ans), f'(frac={majority_fraction:.3f})')
        print('  [post] success =', f"{matches}/{total}", f"rate={success_rate:.3f}")

        return {
            'question': question,
            'answer':   majority_ans,                      
            'majority_fraction': majority_fraction,
            'score':    success_rate,                                                 
            'results':  results,
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

@app.route('/answer', methods=['GET'])
def answer_from_text():





















                                         
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f"[server][answer] Received request for task file: {name}")

               
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

                                             
    valid_items = []
    chats = []
    for item in data:
        txt = (item.get('text') or '').strip()
        q = (item.get('question') or '').strip()
        if not q:
            valid_items.append({
                'text': txt, 'question': q,
                'answer': '', 'majority_fraction': 0.0, 'results': []
            })
            continue
        user_content = (
            "Read the following context and answer the question.\n\n"
            f"Context:\n{txt}\n\n"
            f"Question: {q}"
        )
        if _is_categorical_item(item, q):
            query = user_content + CATEGORICAL_INSTRUCTION
            chats.append([{'role': 'user', 'content': query}])
        else:
            chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': user_content},
            ])
        valid_items.append({'text': txt, 'question': q})

                   
    if chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in chats
            ]
        else:
            prompts = [_chat_to_prompt_fallback(chat) for chat in chats]
    else:
        prompts = []

              
    if prompts:
        print('[server][answer] Gen config:',
              f"n_candidates={sample_params.n}",
              f"temp={sample_params.temperature}",
              f"top_p={sample_params.top_p}",
              f"top_k={sample_params.top_k}",
              f"max_tokens={sample_params.max_tokens}")
        t0 = time.time()
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=False)
        dt = time.time() - t0
        print(f"[server][answer] Generation took {dt:.2f}s for {len(prompts)} prompts.")
    else:
        responses = []

                            
    out_items = []
    r_idx = 0
    for base in valid_items:
        if 'answer' in base:
                                                             
            out_items.append(base)
            continue
        resp = responses[r_idx]
        r_idx += 1
                               
        cand = [extract_boxed_content(out.text) for out in resp.outputs]
                                                           
        cand = [c for c in cand if not _is_invalid_candidate(c)]
        answer_counts = {}
        for res in cand:
            if not res:
                continue
            matched = False
            for exist in list(answer_counts.keys()):
                                      
                if res == exist or ('no ' in res.lower() and 'no ' in exist.lower()):
                    answer_counts[exist] += 1
                    matched = True
                    break
                try:
                    is_match = False
                    m1 = grade_answer_with_timeout(res, exist, timeout=10)
                    if m1 == 'TIMED_OUT':
                        pass
                    elif m1:
                        is_match = True
                    if not is_match:
                        m2 = grade_answer_with_timeout(exist, res, timeout=10)
                        if m2 == 'TIMED_OUT':
                            pass
                        elif m2:
                            is_match = True
                    if is_match:
                        answer_counts[exist] += 1
                        matched = True
                        break
                except Exception as e:
                    print(f"      [answer][cluster] ERROR comparing '{res[:30]}...' and '{exist[:30]}...': {e}")
            if not matched:
                answer_counts[res] = 1

        if answer_counts:
            majority_ans = max(answer_counts, key=answer_counts.get)
            majority_count = answer_counts[majority_ans]
        else:
            majority_ans = ''
            majority_count = 0
        total = len(cand)
        majority_fraction = (majority_count / total) if total > 0 else 0.0
        print('  [answer] question=', _trunc(base.get('question', ''), 100))
        print('  [answer] ctx.len =', len(base.get('text', '') or ''))
        print('  [answer] results =', total, 'cands; top counts=', _top_counts(answer_counts))
        print('  [answer] majority =', _trunc(majority_ans), f'(frac={majority_fraction:.3f})')
        out_items.append({
            'text': base.get('text', ''),
            'question': base.get('question', ''),
            'answer': majority_ans,
            'majority_fraction': majority_fraction,
            'results': cand,
        })

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(out_items, f, indent=2)

    pause_event.clear()
    print(f"[server][answer] Processed {name}, results saved to {out_path}. Resuming idle worker.")
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
