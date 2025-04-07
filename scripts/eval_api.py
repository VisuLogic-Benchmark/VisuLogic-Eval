import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from PIL import Image
import openai
from tqdm import tqdm
import re
import time
import base64
from io import BytesIO
import openai
import numpy as np
import datasets
from datetime import datetime
import random
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import requests.exceptions
import tenacity
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

@dataclass
class EvalConfig:
    model_name: str
    system_prompt: str
    user_prompt: str
    eval_size: int = 100
    random_seed: int = 42
    dataset_path: str = ""
    output_dir: str = "eval_results"
    image_root_path: str = ""
    max_retries: int = 3
    retry_delay: int = 2
    max_image_size: int = 512
    use_dataset_prompt: bool = True
    max_parallel: int = 4
    eval_mode: str = "standard"  # can be "standard" or "interleaved"
    post_metrics: bool = True

@dataclass
class EvalMetrics:
    start_time: str
    end_time: str
    total_duration: float
    average_score: float
    total_samples: int
    successful_samples: int
    config: Dict

class ImageEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.setup_environment()
        if 'deepseek' in self.config.model_name:
            self.client = openai.OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],base_url="https://api.deepseek.com/v1")
        elif 'kimi' in self.config.model_name or 'moonshot' in self.config.model_name:
            self.client = openai.OpenAI(api_key=os.environ['KIMI_API_KEY'],base_url="https://api.moonshot.cn/v1")
        elif 'doubao' in self.config.model_name:
            self.client = openai.OpenAI(api_key=os.environ['DOUBAO_API_KEY'],base_url="https://ark.cn-beijing.volces.com/api/v3")
        else:
            self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        random.seed(config.random_seed)

    def setup_environment(self):
        """Setup necessary environment variables and directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image keeping aspect ratio so that the longest side is max_image_size"""
        width, height = image.size
        if width <= self.config.max_image_size and height <= self.config.max_image_size:
            return image
            
        if width > height:
            new_width = self.config.max_image_size
            new_height = int(height * (self.config.max_image_size / width))
        else:
            new_height = self.config.max_image_size
            new_width = int(width * (self.config.max_image_size / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def image_to_base64(self, image: Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((
            requests.exceptions.RequestException,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(3),
        before_sleep=lambda retry_state: print(f"Retrying after error: {retry_state.outcome.exception()}, "
                                             f"Attempt {retry_state.attempt_number}")
    )


    def get_api_response(self, image, prompt=None) -> str:
        """Get response from API for an image with retry logic"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Resize image before converting to base64
        image = self.resize_image(image)
        img_str = self.image_to_base64(image)
        if prompt is None:
            prompt = self.config.user_prompt

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]}
            ],
        )
        return response.choices[0].message.content

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((
            requests.exceptions.RequestException,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(3),
        before_sleep=lambda retry_state: print(f"Retrying after error: {retry_state.outcome.exception()}, "
                                             f"Attempt {retry_state.attempt_number}")
    )
    def evaluate_item(self, item: Dict) -> Dict:
        """Evaluate a single item"""
        try:
            image_path = os.path.join(self.config.image_root_path, item['image_path'])
            if self.config.use_dataset_prompt:
                prompt = item['text'] + "\n" + self.config.user_prompt
                response = self.get_api_response(image_path, prompt)
            else:
                response = self.get_api_response(image_path)
            
            result_item = {
                **item,
                "model_response": response,
                "evaluation_timestamp": datetime.now().isoformat(),
                "user_prompt": item['text'] + "\n" + self.config.user_prompt  if self.config.use_dataset_prompt else self.config.user_prompt
            }
            
        except Exception as e:
            print(f"Error processing item: {e}")
            result_item = {
                **item,
                "model_response": str(e),
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        
        return result_item
    
    def _create_error_result(self, item: Dict, error_msg: str) -> Dict:
        return {
            **item,
            "model_response": error_msg,
            "error": error_msg,
            "evaluation_timestamp": datetime.now().isoformat()
        }

    def construct_content_list(self, text_img_list):
        content_list = []
        for item in text_img_list:
            try:
                if item.startswith('<img'):
                    image_path = self._extract_image_path(item)
                    with Image.open(os.path.join(self.config.image_root_path,image_path)) as image:
                        image = self.resize_image(image).convert("RGB")
                        img_str = self.image_to_base64(image)
                    
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                else:
                    content_list.append({
                        "type": "text",
                        "text": item
                    })
            except Exception as e:
                print(f"Error processing content item: {e}")
                content_list.append({
                    "type": "text",
                    "text": f"Error processing content: {str(e)}"
                })
        return content_list


    def _extract_image_path(self, img_tag: str) -> str:
        match = re.search(r'src="([^"]+)"', img_tag)
        if not match:
            raise ValueError(f"Invalid image tag format: {img_tag}")
        return match.group(1)

    def evaluate_tv_internleaved(self, text_img):
        try:
            pattern = r'(<img[^>]+>)'
            result = re.split(pattern, text_img)
            result_list = [x for x in result if x.strip()]
            content_list = self.construct_content_list(result_list)

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    # {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": content_list}
                ],
                timeout=30  # 添加超时设置
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in evaluate_tv_internleaved: {e}")
            return f"Error processing request: {str(e)}"

    def evaluate_image_text_mode(self, text_img,bk_text_img):
        # try:
            pattern = r'(<img[^>]+>)'
            match = re.search(r'images\/[a-zA-Z0-9]+\.png', text_img)
            if not match:
                match = re.search(r'images\/[a-zA-Z0-9]+\.png', bk_text_img)
                if not match:
                    raise ValueError(f'bk_text_img {bk_text_img} does not have image')
            image_path = match.group()
            with Image.open(os.path.join(self.config.image_root_path,image_path)) as image:
                image = self.resize_image(image).convert("RGB")
                img_str = self.image_to_base64(image)
            image_require_item = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            }

            text_require_item = {
                "type": "text",
                "text": re.sub(pattern, '', text_img) + '\n' + self.config.user_prompt
            }
            message = [
                    # {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": [
                        image_require_item,text_require_item
                    ]}
                ]

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=message,
                timeout=30  # 添加超时设置
            )
            return response.choices[0].message.content

    def evaluate_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], EvalMetrics]:
        """Evaluate the dataset and output results immediately"""
        start_time = datetime.now()
        
        # Create timestamp directory at the start
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(self.config.output_dir) / timestamp
        base_path.mkdir(parents=True, exist_ok=True)

        if self.config.eval_mode == "interleaved":
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                future_to_item = {}
                for item in dataset[:self.config.eval_size]:
                    try:
                        def process_item(item):
                            response = self.evaluate_tv_internleaved(item['question'])
                            return {
                                **item,
                                "model_response": response,
                                "evaluation_timestamp": datetime.now().isoformat(),
                            }
                        future_to_item[executor.submit(process_item, item)] = item
                    except Exception as e:
                        print(f"Error submitting item: {e}")
                        save_result(self._create_error_result(item, str(e)), base_path)

                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Evaluating"):
                    try:
                        result_item = future.result()
                        save_result(result_item, base_path)
                    except Exception as e:
                        print(f"Error processing future: {e}")
                        item = future_to_item[future]
                        save_result(self._create_error_result(item, str(e)), base_path)
        elif  self.config.eval_mode == 'image-text':
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                future_to_item = {}
                for item in dataset[:self.config.eval_size]:
                    try:
                        def process_item(item):
                            response = self.evaluate_image_text_mode(item['question'],item['question_'])
                            return {
                                **item,
                                "model_response": response,
                                "evaluation_timestamp": datetime.now().isoformat(),
                            }
                        future_to_item[executor.submit(process_item, item)] = item
                    except Exception as e:
                        print(f"Error submitting item: {e}")
                        save_result(self._create_error_result(item, str(e)), base_path)

                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Evaluating"):
                    try:
                        result_item = future.result()
                        save_result(result_item, base_path)
                    except Exception as e:
                        print(f"Error processing future: {e}")
                        item = future_to_item[future]
                        save_result(self._create_error_result(item, str(e)), base_path)
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                future_to_item = {executor.submit(self.evaluate_item, item): item 
                                for item in dataset[:self.config.eval_size]}
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Evaluating"):
                    result_item = future.result()
                    save_result(result_item, base_path)

        end_time = datetime.now()
        
        
        # return scores

def save_result(result: Dict, base_path: Path):
    """Save a single evaluation result"""
    results_file = base_path / "results.jsonl"
    with open(results_file, "a") as f:  # Use append mode
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Image Evaluation Configuration')
    
    # Model configs
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Name of the model to use')
    parser.add_argument('--system_prompt', type=str,
                        default=None,
                        help='System prompt for the model')
    parser.add_argument('--user_prompt', type=str,
                        default='Solve the complex visual logical reasoning problem through step-by-step reasoning. Think about the reasoning process first and answer the question following this format: Answer: \\boxed{$LETTER}.',
                        help='User prompt template')

    parser.add_argument('--use_dataset_prompt', 
                       action='store_true',  # 改用 store_true
                       help='Use prompts from dataset instead of default user prompt')
    
    # Data configs                    
    parser.add_argument('--dataset_path', type=str, default='path/to/BENCHMARK/',
                        help='Path to the dataset file')
    parser.add_argument('--image_root_path', type=str, default='path/to/IMAGE ROOT',
                        help='Root path containing images')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    
    # Runtime configs
    parser.add_argument('--eval_size', type=int, default=1000,
                        help='Size for evaluation')
    parser.add_argument('--max_image_size', type=int, default=512,
                        help='Maximum image size')
    parser.add_argument('--max_parallel', type=int, default=4,
                        help='Max number of parallel workers')
    parser.add_argument('--eval_mode', type=str, default='standard',
                        choices=['standard', 'interleaved','image-text'],
                        help='Evaluation mode: standard or interleaved')
    parser.add_argument('--post_metrics',action="store_true",help="post metrcis with tags,levels...")

    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_args()
    
    config = EvalConfig(
        model_name=args.model_name,
        system_prompt=args.system_prompt, 
        user_prompt=args.user_prompt,
        use_dataset_prompt=args.use_dataset_prompt,
        dataset_path=args.dataset_path,
        image_root_path=args.image_root_path,
        output_dir=args.output_dir,
        eval_size=args.eval_size,
        max_image_size=args.max_image_size,
        max_parallel=args.max_parallel,
        eval_mode=args.eval_mode
    )
    # Setup evaluator
    evaluator = ImageEvaluator(config)
    
    # Load dataset
    dataset = pd.read_json(config.dataset_path, lines=True).to_dict(orient="records")
    
    # Run evaluation
    evaluator.evaluate_dataset(dataset)

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = "sk-"
    os.environ['KIMI_API_KEY'] = 'sk-'
    os.environ['DEEPSEEK_API_KEY'] = 'sk-'
    os.environ['BOYUE_API_KEY'] = 'sk-'
    main()
