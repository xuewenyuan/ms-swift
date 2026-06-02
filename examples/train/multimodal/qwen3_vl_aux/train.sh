QWEN3VL_AUX_CONFIG_PATH=examples/train/multimodal/qwen3_vl_aux/config.example.json \
QWEN3VL_AUX_HEAD_LR=5e-4 \
QWEN3VL_AUX_LM_WEIGHT=0 \
QWEN3VL_AUX_DISABLE_LORA=1 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset /path/to/your_train.jsonl \
    --external_plugins examples/train/multimodal/qwen3_vl_aux/plugin.py \
    --torch_dtype bfloat16 \
    --train_type qwen3vl_aux \
    --loss_type qwen3vl_aux \
    --optimizer qwen3vl_aux \
    --remove_unused_columns false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --output_dir output/qwen3_vl_aux
