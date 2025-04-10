#export CUDA_VISIBLE_DEVICES=2,3
gpu_num=${1:1}
gpu_id=${2:0}
OUTPUT_DIR=/PATH/to/your/output


python eval_qwen2vl_7b.py \
    --local_model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_num $gpu_num \
    --gpu_id $gpu_id \
    --output_dir ${OUTPUT_DIR} \
