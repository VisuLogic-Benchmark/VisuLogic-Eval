from models.doubao_api import DoubaoAPIModel
from models.openai_api import OpenAIAPIModel
from models.kimi_api import MoonshotAPIModel
from models.qwenvl_model import QwenVisionModel
from models.internvl_model import InternVLModel

def load_model(args):
    if args.model_path in ["doubao-vision-pro-32k","doubao-vision-lite-32k","doubao-1.5-vision-pro-32k"]:
        model = DoubaoAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif args.model_path in ['gpt-4o','gpt-4o-mini']:
        model = OpenAIAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif args.model_path in ['kimi-latest','moonshot-v1-8k-vision-preview','moonshot-v1-32k-vision-preview','moonshot-v1-128k-vision-preview']:
        model = MoonshotAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif 'qwen' in args.model_path.lower() and 'vl' in args.model_path.lower():
        model = QwenVisionModel(model_path = args.model_path
                                user_prompt = args.user_prompt)
    elif 'internvl' in args.model_path.lower():
        model = InternVLModel(model_path = args.model_path
                                user_prompt = args.user_prompt)
    return model