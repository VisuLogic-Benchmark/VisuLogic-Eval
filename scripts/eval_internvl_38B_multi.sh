#export CUDA_VISIBLE_DEVICES=2,3
gpu_num=$1
gpu_id=$2
OUTPUT_DIR=/PATH/to/your/output

python eval_internvl_38B.py \
    --local_model_path OpenGVLab/InternVL2_5-38B \
    --gpu_num $gpu_num \
    --gpu_id $gpu_id \
    --output_dir ${OUTPUT_DIR}
