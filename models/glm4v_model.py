from models.base_model import BaseModel
from PIL import Image
from typing import Any, Dict
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


class GLM4VModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None):
        """
        Initialize the GLM4V Model.
        Args:
            model_path: Path to the model.
            user_prompt: user_prompt.
        """
        self.user_prompt = user_prompt
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


        self.model = model
        self.tokenizer = tokenizer


    @property
    def name(self) -> str:
        return "glm4v"


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
        query = question +'\n'+self.user_prompt
        image = Image.open(image_path).convert('RGB')
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode
        # print(query)
        # exit()
        inputs = inputs.to(self.model.device)
        gen_kwargs = {"max_length": 1280, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        #print(tokenizer.decode(outputs[0]))
        return self.tokenizer.decode(outputs[0])