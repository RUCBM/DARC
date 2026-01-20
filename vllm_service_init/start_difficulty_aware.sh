model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1


MAX_MODEL_LEN=${MAX_MODEL_LEN:-16000}


CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server_difficulty_aware.py --port 5000 --model_path $model_path --max_model_len ${MAX_MODEL_LEN} &
CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server_difficulty_aware.py --port 5001 --model_path $model_path --max_model_len ${MAX_MODEL_LEN} &
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server_difficulty_aware.py --port 5002 --model_path $model_path --max_model_len ${MAX_MODEL_LEN} &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server_difficulty_aware.py --port 5003 --model_path $model_path --max_model_len ${MAX_MODEL_LEN} &