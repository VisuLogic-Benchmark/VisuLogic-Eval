
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
from transformers import AutoModelForCausalLM



def load_model(model_path,gpu_id):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda(gpu_id)
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model,text_tokenizer,visual_tokenizer

def get_model_response(model,text_tokenizer,visual_tokenizer,image_path,question,user_prompt):
    images = [Image.open(image_path)]
    max_partition = 9
    text = question +'\n'+user_prompt
    query = f'<image>\n{text}'

    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        #print(f'Output:\n{output}')
        return output



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
    model,text_tokenizer,visual_tokenizer = load_model(model_path=args.local_model_path,gpu_id=args.gpu_id)
    for item in tqdm(dataset):

        image_path = item["image"]

        model_response = get_model_response(model=model,text_tokenizer=text_tokenizer,visual_tokenizer=visual_tokenizer,image_path=image_path,
                                            question=item["question"],user_prompt=args.user_prompt)
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