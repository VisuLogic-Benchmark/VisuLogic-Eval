mkdir -p outputs/
python evaluation/eval_model.py \
    --input_file path/to/data.jsonl \
    --output_file outputs/output_file.jsonl \
    --model_path OpenGVLab/InternVL2_5-78B \
    --judge_api_key sk-xxx