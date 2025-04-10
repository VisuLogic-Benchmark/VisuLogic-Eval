# VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models

**A Chanllenging Visual-centric Benchmark for Evaluating Multimodal Reasoning in MLLMs!**

Paper, training datasets, training codes and model checkpoints are coming!

For more details, please refer to the project page with dataset exploration and visualization tools: [https://visulogic-benchmark.github.io/VisuLogic/](https://visulogic-benchmark.github.io/VisuLogic/).

# VisuLogic Benchmark

[**🌐 Homepage**](https://visulogic-benchmark.github.io/VisuLogic) | [**🏆 Leaderboard**(coming soon)](https://visulogic-benchmark.github.io/VisuLogic/) |



## 🔔News

- **🔥[2025-04-08] Release the benchmark and the codes! 🚀**
## To-do
- [x] Release the benchmark dataset and eval codes
- [ ] Release training codes
- [ ] Release the paper
- [ ] Release the training dataset
- [ ] Release model ckpts


![Overview](assets/overview4.png)

## 🌟 Key Features

- 🚀 **Visuo-Logical Challenge**  
  The first benchmark to integrate **visual perception** with **logical reasoning**, enabling authentic multimodal evaluation.
  
- 🛠️ **Rigorous Design**  
  Includes **1,000 meticulously curated questions**, spanning **6 domains** and **23 subcategories**, for comprehensive performance evaluation.
  
- 📝 **Anti-Linguistic Shortcut**  
  Designed to avoid linguistic biases, ensuring tasks rely on **genuine visual reasoning** rather than shortcuts.

- 📊 **Human-Aligned Evaluation**  
  - **Human Accuracy**: >50.0%  
  - **State-of-the-Art (SOTA) MLLMs Accuracy**: <30%

## Benchmark Data

For more detailed information, please refer to our Hugging Face datasets:

- [**🤗 VisuLogic Dataset**](https://huggingface.co/datasets/VisuLogic/VisuLogic)

## Evaluation
Firstly you should clone our repo and prepare the packages

```bash
# Clone repository
git clone https://github.com/VisuLogic-Benchmark/VisuLogic-Eval.git

# Install dependencies
pip install -r requirements.txt
```

Navigate to the `scripts` directory containing preconfigured evaluation pipelines. Run the corresponding evaluation script with specific parameters. For Qwen2.5-VL-Instruct:
```bash
# Run evaluation for specific model (e.g. Qwen2.5-VL-Instruct)
cd scripts
bash eval_qwen2.5vl_7b_multi.sh 
```


## Contact
- Jiahao Wang: wjhwdscience@stu.xjtu.edu.cn
- Weiye Xu: ustcxwy0271@mail.ustc.edu.cn

## Citation

**BibTeX:**
```bibtex
@misc{visulogic,
    title        = {VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models},
    author       = {VisuLogic-Benchmark},
    howpublished = {\url{https://github.com/VisuLogic-Benchmark/VisuLogic-Eval}},
    year         = {2025},
    note         = {Accessed: 2025-04-08}
}
```
