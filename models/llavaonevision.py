from models.base_model import BaseModel
from typing import Any, Dict, List
import torch
from transformers import AutoProcessor
from PIL import Image
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


class LlavaOnevisionModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None):
        """
        Initialize the Llava-Onevision Model.
        Args:
            model_path: Path to the model.
            user_prompt: user_prompt.
        """
        model_name = "llava_qwen"
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map="auto")  # Add any other thing you want to pass in llava_model_args

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.user_prompt = user_prompt


    @property
    def name(self) -> str:
        return "llavaonevision"


    def predict(self, input_data: Dict) -> str:
        """
        Model prediction interface
        Args:
            input_data: Dictionary containing image path and question
        Returns:
            str: Model prediction result
        """
        image = Image.open(input_data['image_path'])
        question = input_data['text']
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + question +'\n'+self.user_prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        image_sizes = [image.size]


        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=1024,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        #print(text_outputs)
        return text_outputs[0]