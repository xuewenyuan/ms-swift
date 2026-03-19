# 1) Start rollout server first (use dedicated GPU(s), not training GPUs):
# CUDA_VISIBLE_DEVICES=6 \
# swift rollout \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --vllm_max_model_len 3072 \
#     --server_port 8000
#
# 2) Prepare OPSD dataset (jsonl) with "<reference>" placeholder:
# {
#   "messages": [{"role": "user", "content": "Question... <reference>"}],
#   "reference": "Reference content..."
# }
# Trainer behavior:
# - student path: remove "<reference>"
# - teacher path: replace "<reference>" with `reference` value

nproc_per_node=2
dataset_path=./data/opsd_train.jsonl

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
python train.py \
    --pipeline rlhf \
    --rlhf_type opsd \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset "$dataset_path" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 512 \
    --beta 0.5 \
    --lmbda 1 \
    --output_dir output/opsd_vllm_server \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000
