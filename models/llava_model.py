from models.base_model import BaseModel
from PIL import Image
from typing import Any, Dict
import torch
from datetime import datetime
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


class LlavaModel(BaseModel):
    def __init__(self, model_path: str,user_prompt: str = None):
        """
        Initialize the Llava Model.
        Args:
            model_path: Path to the model.
            user_prompt: user_prompt.
        """
        self.user_prompt = user_prompt
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor


    @property
    def name(self) -> str:
        return "llava"


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
        if self.model.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question +'\n'+self.user_prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question +'\n'+self.user_prompt
        # print(qs)
        # exit()
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.model_config)[0]
        image_tensor = image_tensor.unsqueeze(0)
        image_sizes = [image.size]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=self.model.device, non_blocking=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=self.model.device, non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs