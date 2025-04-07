export CUDA_VISIBLE_DEVICES=0
python eval_llava.py \
    --local_model_path liuhaotian/llava-v1.5-7b \
    --api_key xxxxx \
    --gpu_id 0 \
    --output_dir llava-v1_5