export CUDA_VISIBLE_DEVICES=1
python eval_sharegpt4v.py \
    --local_model_path Lin-Chen/ShareGPT4V-7B \
    --api_key xxxxx \
    --gpu_id 0 \
    --output_dir sharegpt4v