from models.base_model import BaseModel
from PIL import Image
from typing import Any, Dict
import torch
from modelscope import AutoConfig, AutoModel
from modelscope import AutoTokenizer

class mPLUGModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None):
        """
        Initialize the mPLUG Model.
        Args:
            model_path: Path to the model.
            user_prompt: user_prompt.
        """
        self.user_prompt = user_prompt
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(config)
        model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True)
        _ = model.eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor


    @property
    def name(self) -> str:
        return "mPLUG"


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
        messages = [
            {"role": "user", "content": "<|image|>\n"+question +'\n'+self.user_prompt},
            {"role": "assistant", "content": ""}
        ]

        inputs = self.processor(messages, images=[image], videos=None)
        # print(messages)
        # exit()
        inputs.to(self.model.device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens':1024,
            'decode_text':True,
        })
        g = self.model.generate(**inputs)
        #print(g)
        return g[0]