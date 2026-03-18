nproc_per_node=2
dataset_path=./data/opsd_train.jsonl

# Expected sample format (jsonl):
# {
#   "messages": [{"role": "user", "content": "Question... <reference>"}],
#   "reference": "Reference content..."
# }
# OPSD trainer behavior:
# - student path: remove "<reference>" from user content
# - teacher path: replace "<reference>" with value from "reference" field

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
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
    --output_dir output/opsd \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn
