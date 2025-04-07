
from PIL import Image
import requests
import copy
import torch
import json

import sys
import warnings
import argparse
import jsonlines
from datetime import datetime
import os
import time
from openai import OpenAI
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import math
import random


def get_index(num_gpus,chunk_id,data_length):
    chunk_size = data_length / num_gpus

    start = int(chunk_id * chunk_size)
    end = int((chunk_id + 1)* chunk_size)
    
    return start , end

def load_model(model_path,gpu_id):
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    #default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    return model, processor

def get_model_response(model,processor,image_path,question,user_prompt):
    # test.py
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question +'\n'+user_prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    #print(output_text)
    return output_text[0]



def load_dataset(file_path):
    data = []
    with open(file_path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def judge_response(response_content: str, label: str,client,judge_model_name) -> bool:
    """Judge if the response matches the label with retry logic"""
    while True:
        try:
            response = client.chat.completions.create(
                model=judge_model_name,
                messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                    {"role": "user", "content": f"Check if the option in the solution matches the label. Answer only Yes or No.If solution have multi options or has no clear option just answer No.\nSolution: {response_content}\nLabel: {label}"}
                ],
            )
            return "yes" in response.choices[0].message.content.lower(), response.choices[0].message.content.lower()
        except:
            time.sleep(1)
            print("requeset failed, retrying...")

def parse_args():
    parser = argparse.ArgumentParser(description='Image Evaluation Configuration')
    
    # Model configs
    parser.add_argument('--local_model_path', type=str, required=True,
                        help='Name of the model to use')
    parser.add_argument('--judge_model_name', type=str, default='gpt-4o-mini',
                        help='Name of the judge model to use')
    parser.add_argument('--api_key', type=str, default='',
                        help='api_key')
    parser.add_argument('--base_url', type=str, default='',
                        help='base_url')
    parser.add_argument('--user_prompt', type=str,
                        default='Solve the complex visual logical reasoning problem through step-by-step reasoning. Think about the reasoning process first and answer the question following this format: Answer: \\boxed{$LETTER}.',
                        help='User prompt template')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='gpu_num')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu_id')
    
    # Data configs                    
    parser.add_argument('--dataset_path', type=str, default='benchmark_en.jsonl',
                        help='Path to the dataset file')
    parser.add_argument('--image_root_path', type=str, default='',
                        help='Root path containing images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()
    return args

def evaluate(args):
    dataset = load_dataset(args.dataset_path)
    start,end = get_index(args.gpu_num,args.gpu_id,len(dataset))
    dataset = dataset[start:end]
    save_jsonl = os.path.join(args.output_dir,"results.jsonl")
    api_keys = []
    # 多个api_keys 可以存入该路径
    with open("key_pool.txt", "r") as file:
        for l in file:
            l = l.strip()  # 移除行末尾的换行符
            api_keys.append(l)
    #judeg_client = OpenAI(api_key=args.api_key,base_url=args.base_url)
    model,processor = load_model(model_path=args.local_model_path,gpu_id=args.gpu_id)
    for item in tqdm(dataset):

        image_path = item["image"]
        judeg_client = OpenAI(api_key=random.choice(api_keys),base_url=args.base_url)
        model_response = get_model_response(model=model,processor=processor,image_path=image_path,
                                            question=item["question"],user_prompt=args.user_prompt)
        #is_correct , judge_output = judge_response(model_response, item['answer'],judeg_client,judge_model_name=args.judge_model_name)
        #score = 1.0 if is_correct else 0.0
        save_item = {
                **item,
                "model_response": model_response,
                # "score": score,
                # "judge_output": judge_output,
                "evaluation_timestamp": datetime.now().isoformat(),
            }
        with jsonlines.open(save_jsonl, "a") as f:
            f.write(save_item)



if __name__ == "__main__":
    args = parse_args()
    evaluate(args=args)