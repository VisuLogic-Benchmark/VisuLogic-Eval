from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

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

warnings.filterwarnings("ignore")

def load_model(model_path,gpu_id):
    model_name = "llava_qwen"
    device_map = f"cuda:{gpu_id}"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    return model,tokenizer,image_processor

def get_model_response(model,tokenizer,image_path,question,user_prompt,image_processor,gpu_id):
    image = Image.open(image_path)
    device=f"cuda:{gpu_id}"
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + question +'\n'+user_prompt
    # print(question)
    # exit()
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=1024,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    #print(text_outputs)
    return text_outputs[0]

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
    save_jsonl = os.path.join(args.output_dir,"results.jsonl")
    judeg_client = OpenAI(api_key=args.api_key,base_url=args.base_url)
    model,tokenizer,image_processor = load_model(model_path=args.local_model_path,gpu_id=args.gpu_id)
    for item in tqdm(dataset):
        image_path = item["image"]

        model_response = get_model_response(model=model,tokenizer=tokenizer,image_path=image_path,
                                            question=item["question"],user_prompt=args.user_prompt,
                                            image_processor=image_processor,gpu_id=args.gpu_id)
        is_correct , judge_output = judge_response(model_response, item['answer'],judeg_client,judge_model_name=args.judge_model_name)
        score = 1.0 if is_correct else 0.0
        save_item = {
                **item,
                "model_response": model_response,
                "score": score,
                "judge_output": judge_output,
                "evaluation_timestamp": datetime.now().isoformat(),
            }
        with jsonlines.open(save_jsonl, "a") as f:
            f.write(save_item)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args=args)