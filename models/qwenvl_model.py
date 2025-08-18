from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import re
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import os
from qwen_vl_utils import process_vision_info
from models.base_model import BaseModel

class QwenVisionModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None, max_image_size: int = -1):
        """
        Initialize the Qwen Vision Model.
        Args:
            model_path: Path to the model.
            model_version: Model version ("2.0" or "2.5").
        """
        if "2.5" in model_path or "2_5" in model_path:
            model_version = 2.5
        else:
            model_version = 2
        self.model_path = model_path
        self.model_version = model_version
        self.user_prompt = user_prompt
        self.max_image_size = max_image_size
        # load model and processor
        # ModelClass = Qwen2_5_VLForConditionalGeneration if model_version == "2.5" else Qwen2VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    @property
    def name(self) -> str:
        return self.model.config._name_or_path

    def _prepare_input(self, image_path: str,text: str) -> tuple:
        input_image = Image.open(image_path)
        input_text = '<image>\n' + text + self.user_prompt
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
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        return text, image_inputs, video_inputs

    def predict(self, input_data: Dict) -> str:
        """
        Model prediction interface
        Args:
            input_data: Dictionary containing image path and question
        Returns:
            str: Model prediction result
        """
        # Extract image path from input data
        # Prepare inputs
        text, image_inputs, video_inputs = self._prepare_input(
            image_path=input_data['image_path'],
            text=input_data['text']
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate predictions
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,  
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""