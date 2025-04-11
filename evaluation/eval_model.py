import argparse
import json
import statistics
import os
import re
import time
from openai import OpenAI
from tqdm import tqdm
from models import load_model
Option_list = ['A','B','C','D']

class ModelEvaluator:
    def __init__(self, args):
        self.model = load_model(args)
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.score_field = getattr(args, 'score_field', 'score')
        self.tag_field = getattr(args, 'tag_field', 'tag')
        self.verbose = getattr(args, 'verbose', False)
        self.api_key = getattr(args, 'api_key', '')
        self.base_url = getattr(args, 'base_url', '')
        self.judge_model_name = getattr(args, 'judge_model_name', 'gpt-4-mini')
        self.judge_client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None
        
        # Define tag order
        self.tag_order = [
            "Quantitative Reasoning",
            "Spatial Reasoning",
            "Positional Reasoning",
            "Attribute Reasoning",
            "Stylistic Reasoning",
            "Other"
        ]

    def extract_last_boxed_content(self, text):
        stack = []
        last_boxed_content = None
        text = str(text)
        if len(text) < 3:
            return text
        
        pattern = re.finditer(r'\\boxed\{|[^\\]\}', text)
        
        try:
            for match in pattern:
                if match.group().endswith(r'\boxed{'):
                    stack.append(match.end())
                elif match.group().endswith('}') and stack:
                    start = stack.pop()
                    if not stack:
                        last_boxed_content = text[start:match.start() + 1]
            
            if last_boxed_content:
                latex_commands = [r'\text{', r'\rm{', r'\mathbf{', '$']
                for cmd in latex_commands:
                    last_boxed_content = last_boxed_content.replace(cmd, '')
                last_boxed_content = last_boxed_content.replace('}', '')
                
                if ("LETTER".lower() in last_boxed_content.lower() or 
                    'or' in last_boxed_content or 
                    len(last_boxed_content) > 2):
                    last_boxed_content = text
                    
        except Exception:
            last_boxed_content = text
            
        return 'N' if last_boxed_content==None else last_boxed_content

    def extract_lang_content(self, ans):
        ans = str(ans)
        ans = ans.replace("<|endoftext|>","")
        for c in Option_list:
            if ans.endswith(f" {c}.") or ans.endswith(f" ({c}).") or ans.startswith(f"{c}\n") or ans.startswith(f"({c})\n") or ans.startswith(f"({c}) {c}\n"):
                return c
        
        lower_ans = ans.lower()
        for flag in ["answer:",'the final answer is:', 'the answer is option:', 'the answer is:',
                    'the correct answer is option:','the correct answer is:', 'the answer should be:',
                    'the final answer is', 'the answer is option', 'the answer is',
                    'the correct answer is option','the correct answer is','the answer should be']:
            if flag in lower_ans:
                lower_ans = lower_ans.split(flag)[-1].strip()
                lower_ans = lower_ans.split('\n')[0].split('.')[0]
                upper_ans = lower_ans.upper()
                if upper_ans in Option_list:
                    return upper_ans
        
        return ans

    def extract_gpt(self, ans):
        while True:
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful and precise assistant for extract the final option in the answer."},
                        {"role": "user", "content": f"Extract the option in the solution. Just answer only capital letters. If solution have multi options or has no clear option just answer None.\nSolution: {ans}\n"}
                    ],
                )
                extracted_answer = response.choices[0].message.content.strip()
                extracted_answer = extracted_answer.split('\n')[0].split('.')[0]
                return 'N' if "none" in extracted_answer.lower() else extracted_answer
            except:
                time.sleep(1)
                print("request failed, retrying...")

    def extract_answer_v1(self, ans):
        if self.extract_last_boxed_content(ans).strip() in Option_list:
            return self.extract_last_boxed_content(ans).strip(), "box"
        elif self.extract_lang_content(ans) in Option_list:
            return self.extract_lang_content(ans), "lang"
        else:
            print("Rule extraction failed. The current answer is: ", ans)
            return self.extract_gpt(ans), "gpt"

    def load_data(self):
        """Load jsonl data"""
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def write_result(self, result):
        """Write a single result"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def calculate_scores(self, eval_results):
        scores = {tag: [] for tag in self.tag_order}
        all_scores = []

        for result in eval_results:
            score = 1.0 if result['answer'].lower() == result['extracted_answer'].lower() else 0.0
            tag = result[self.tag_field]
            
            all_scores.append(score)
            if tag in self.tag_order:
                scores[tag].append(score)
            elif self.verbose:
                print(f"unknown tag '{tag}'")

        # Calculate statistics
        tag_scores = {}
        for tag in self.tag_order:
            if scores[tag]:
                avg = statistics.mean(scores[tag])
                tag_scores[tag] = round(avg, 3) * 100
            else:
                tag_scores[tag] = 0.00
        
        overall_acc = round(statistics.mean(all_scores) * 100, 2) if all_scores else 0.00
        
        return tag_scores, overall_acc

    def evaluate(self):
        """Main evaluation function"""
        eval_data = self.load_data()
        open(self.output_file, 'w').close()
        
        eval_results = []
        for item in tqdm(eval_data, desc="Evaluating"):
            # Model inference
            model_response = self.model.predict(item['image_path'], item['question'])
            
            # Extract answer
            extracted_answer, extractor = self.extract_answer_v1(model_response)
            
            # Construct output result
            result = {
                **item,
                'model_response': model_response,
                'extracted_answer': extracted_answer,
                'extractor': extractor
            }
            
            eval_results.append(result)
            self.write_result(result)

        # Calculate and output scores
        tag_scores, overall_acc = self.calculate_scores(eval_results)
        
        # Generate output formats
        scores_str = " & ".join([str(f'{tag_scores[tag]:.1f}') for tag in self.tag_order])
        excel_scores_str = ";".join([str(f'{tag_scores[tag]:.1f}') for tag in self.tag_order])
        
        model_name = self.get_model_name()
        latex_output = f"{model_name} & {overall_acc:.1f} & {scores_str} \\\\"
        excel_output = f"{overall_acc:.1f};{excel_scores_str}"
        
        print(latex_output)
        print(excel_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input jsonl file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output jsonl file path')
    parser.add_argument('--model_type', type=str, required=True, help='Model type to load')
    parser.add_argument('--model_path', type=str, required=True, help='Model path or name')
    parser.add_argument('--score_field', type=str, default='score', help='Score field name')
    parser.add_argument('--tag_field', type=str, default='tag', help='Tag field name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    parser.add_argument('--api_key', type=str, default='', help='API key')
    parser.add_argument('--base_url', type=str, default='', help='Base URL')
    parser.add_argument('--user_prompt', type=str, default='', help='user prompt')
    parser.add_argument('--judge_model_name', type=str, default='gpt-4-mini', help='Judge model name')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args)
    evaluator.evaluate()

if __name__ == '__main__':
    main()