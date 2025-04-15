mkdir -p outputs/
python evaluation/eval_model.py \
    --input_file path/to/data.jsonl \
    --output_file outputs/output_file.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --judge_api_key sk-xxx