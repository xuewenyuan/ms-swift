CKPT_DIR=${CKPT_DIR:-output/qwen3_vl_aux/checkpoint-100}
OUTPUT_DIR=${OUTPUT_DIR:-${CKPT_DIR}-merged-lm}

QWEN3VL_AUX_LOAD_HEADS=false \
swift export \
    --external_plugins examples/train/multimodal/qwen3_vl_aux/plugin.py \
    --adapters "$CKPT_DIR" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR" \
    --exist_ok true
