from models.base_model import BaseModel
import openai
from PIL import Image
import base64
from io import BytesIO
from typing import Dict

class MoonshotAPIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.moonshot.cn/v1", user_prompt: str = None, max_image_size: int = -1):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.max_image_size = max_image_size
        
        # Initialize API client
        if base_url:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return self.model_name

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize the image while maintaining the aspect ratio"""
        width, height = image.size
        if width <= self.max_image_size and height <= self.max_image_size:
            return image
            
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL image to a base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def predict(self, input_data: Dict) -> str:
        """Process input and get model prediction"""
        try:
            return self._predict_image_text(input_data)
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def _predict_image_text(self, input_data: Dict) -> str:
        """Image-text mode prediction"""
        image = Image.open(input_data['image_path'])
        if self.max_image_size > 0:
            image = self.resize_image(image).convert("RGB")
        img_str = self.image_to_base64(image)

        text = input_data['text'] + '\n' + self.user_prompt
        
        message = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                {"type": "text", "text": text}
            ]}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message
        )
        return response.choices[0].message.content
