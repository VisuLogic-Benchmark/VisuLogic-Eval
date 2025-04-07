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
from transformers import Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def evaluate_dataset(model, processor, dataset_path, output_path_dir,):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(output_path_dir) / timestamp
    base_path.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(base_path,'results.jsonl')
    print(output_path)
    model_name = model.config._name_or_path
    with open(base_path / "model_name.txt", "w") as f:
        f.write(model_name)
        
    with open(dataset_path,'r') as dataf,open(output_path, 'w') as outf:
        lines = dataf.readlines()
        for line in tqdm(lines):
            data_item = json.loads(line)
            match = re.search(r'images/\w+\.png', data_item['question_'])
            if match:
                image_path= match.group()  # 输出: images/18c0a906e53df5e.png
            else:
                raise NotImplementedError
            image_path = os.path.join("path/to/IMAGE ROOT",image_path)
            input_image = Image.open(image_path)
            clean_text = re.sub(r'<img\b[^>]*>', '', data_item['question'])  # 删除 <img> 标签
            input_text = '<image>\n' + clean_text + "\nSolve the complex visual logical reasoning problem through step-by-step reasoning. Think about the reasoning process first and answer the question following this format: Answer: \\boxed{$LETTER}."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": input_image,
                        },
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
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
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=8192)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return_item = {
                **data_item,
                'model_response': output_text
            }
            outf.write(json.dumps(return_item,ensure_ascii=False) + '\n')
            outf.flush() 



if __name__ == "__main__":
    path = os.environ.get("MODEL_PATH")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(path)
    
    evaluate_dataset(model,processor,
                        "/path/to/benchmark.jsonl",
                        'eval_results')

