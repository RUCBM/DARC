#!/bin/bash

export INJECT_EXTRA_INFO_TO_GROUND_TRUTH=0
export VLLM_DISABLE_COMPILE_CACHE=1

solver_model_path=$1
solver_train_file=$2
experiment_name=$3

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.shuffle=false \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=512 \
    data.train_files=${solver_train_file} \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    trainer.self_vote_ratio_use_all=true \
    trainer.dump_rollout_n=64 \
    trainer.dump_rollout_every=1 \
    trainer.dump_rollout_path=${STORAGE_PATH}/models/${experiment_name}/rollouts.jsonl \
    trainer.total_epochs=1 \
    trainer.max_steps=512 \
    trainer.save_limit=10 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    data.train_prompt_key=prompt\
    data.train_answer_key=reward_model \
    data.val_prompt_key=problem \
    data.val_answer_key=answer \
    trainer.val_freq=4 \
    trainer.save_freq=16 \
    algorithm.kl_coef=0 \
    worker.rollout.n=8 \
    algorithm.norm_adv_by_std_in_grpo=false \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.max_num_batched_tokens=15000 \
    worker.actor.ulysses_sequence_parallel_size=4 \
    worker.reward.reward_function=./examples/reward_function/difficulty_aware_solver.py:compute_score \
    worker.reward.reward_function_kwargs.solver_label_mode=self_vote \
    worker.reward.reward_function_kwargs.label_prompt_key=text_prompt \
    worker.reward.reward_function_kwargs.label_vote_threshold=0.3 \
    worker.reward.reward_function_kwargs.label_n=8 \
    worker.reward.reward_function_kwargs.label_temperature=1.0 \
    worker.reward.reward_function_kwargs.label_top_p=0.95
