from models.base_model import BaseModel
from PIL import Image
from typing import Any, Dict
import torch
from transformers import AutoModel, AutoTokenizer

class MiniCPMOModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None):
        """
        Initialize the MiniCPM Model.
        Args:
            model_path: Path to the model.
            user_prompt: user_prompt.
        """
        self.user_prompt = user_prompt
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False
        )

        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.model = model
        self.tokenizer = tokenizer


    @property
    def name(self) -> str:
        return "minicpm"


    def predict(self, input_data: Dict) -> str:
        """
        Model prediction interface
        Args:
            input_data: Dictionary containing image path and question
        Returns:
            str: Model prediction result
        """
        question = input_data['text']
        image_path = input_data['image_path']
        image = Image.open(image_path).convert('RGB')
        prompt = question +'\n'+self.user_prompt
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = self.model.chat(
            #image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return res