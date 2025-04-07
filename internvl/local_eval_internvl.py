import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import deepspeed
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
import os
import sys
import os
import socket
import subprocess
from datetime import timedelta
import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
import time
import base64
from io import BytesIO
import numpy as np
import datasets
from datetime import datetime
import random
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import requests.exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import openai
import pandas as pd
import random
import io

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def save_result(result: Dict, base_path: Path):
    """Save a single evaluation result"""
    results_file = base_path / "results.jsonl"
    with open(results_file, "a") as f:  # Use append mode
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def evaluate_dataset(model, tokenizer, dataset_path, output_path_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(output_path_dir) / timestamp
    base_path.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(base_path,'results.jsonl')
    print(output_path)
    model_name = model.config._name_or_path
    with open(base_path / "model_name.txt", "w") as f:
        f.write(model_name)
    with open(dataset_path,'r') as dataf:
        outf = open(output_path , 'w')
        lines = dataf.readlines()
        for line in tqdm(lines):
            data_item = json.loads(line)
            match = re.search(r'images/\w+\.png', data_item['question_'])
            if match:
                image_path= match.group()  # 输出: images/18c0a906e53df5e.png
            else:
                raise NotImplementedError
            image_path = os.path.join("path/to/IMAGE ROOT",image_path)
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            clean_text = re.sub(r'<img\b[^>]*>', '', data_item['question'])  # 删除 <img> 标签
            input_text = '<image>\n' + clean_text + "\nSolve the complex visual logical reasoning problem through step-by-step reasoning. Think about the reasoning process first and answer the question following this format: Answer: \\boxed{$LETTER}."
            generation_config = dict(max_new_tokens=8192, do_sample=True)
            response = model.chat(tokenizer, pixel_values, input_text, generation_config)
            return_item = {
                **data_item,
                'model_response': response
            }

            outf.write(json.dumps(return_item,ensure_ascii=False) + '\n')
            outf.flush() 



if __name__ == "__main__":
    path = os.environ.get("MODEL_PATH")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'
    
    evaluate_dataset(model,tokenizer,
                        "path/to/benchmark.jsonl",
                        'eval_results')

