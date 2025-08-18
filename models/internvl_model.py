from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os
from tqdm import tqdm
import re
from datetime import datetime
import json
from models.base_model import BaseModel

class InternVLModel(BaseModel):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(self, model_path: str, user_prompt: str = None, input_size: int = 448, max_num: int = 12):
        self.model_path = model_path
        self.input_size = input_size
        self.max_num = max_num
        self.user_prompt = user_prompt
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='auto'
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        
        # Create image transformation
        self.transform = self._build_transform()

    @property
    def name(self) -> str:
        return self.model.config._name_or_path

    def _build_transform(self):
        """Build image preprocessing transformation"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: set, width: int, height: int) -> tuple:
        """Find the closest aspect ratio"""
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
                if area > 0.5 * self.input_size * self.input_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image, min_num: int = 1) -> List[Image.Image]:
        """Dynamically preprocess the image"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, self.max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= self.max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height)

        # Calculate target width and height
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize and split the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        # Add thumbnail
        if len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess the image"""
        image = Image.open(image_path).convert('RGB')
        images = self._dynamic_preprocess(image)
        pixel_values = [self.transform(image) for image in images]
        return torch.stack(pixel_values)

    def predict(self, input_data: Dict) -> str:
        """
        Model prediction interface
        Args:
            input_data: Dictionary containing image path and question
        Returns:
            str: Model prediction result
        """   
        image_path = input_data['image_path']

        pixel_values = self._load_image(image_path).to(torch.bfloat16).cuda()

        text = input_data['text']
        input_text =  text + "\n" + self.user_prompt
        
        # Generate configuration
        generation_config = dict(max_new_tokens=8192, do_sample=True)
        
        # Get model response
        response = self.model.chat(self.tokenizer, pixel_values, input_text, generation_config)
        
        return response