# VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models

**A Chanllenging Visual-centric Benchmark for Evaluating Multimodal Reasoning in MLLMs!**

Paper, training datasets, training codes and model checkpoints are coming!

For more details, please refer to the project page with dataset exploration and visualization tools: [https://visulogic.github.io/](https://visulogic.github.io/).

# VisuLogic Benchmark

[**🌐 Homepage**](https://visulogic.github.io/) | [**🏆 Leaderboard**(coming soon)](https://visulogic.github.io/) |



## 🔔News

- **🔥[2025-04-08] Release the benchmark and the codes! 🚀**
## To-do
- [x] Release the benchmark dataset and eval codes
- [ ] Release training codes
- [ ] Release the paper
- [ ] Release the training dataset
- [ ] Release model ckpts



## 🌟 Key Features
- **Visuo-Logical Challenge**: First benchmark integrating visual perception with logical reasoning for authentic multimodal evaluation
- **Rigorous Design**: 1,000 meticulously curated questions across 6 domains and 24 subcategories
- **Anti-Linguistic Shortcut**: Spatial reasoning tasks requiring genuine multimodal understanding
- **Human-Aligned Evaluation**:  
  + Human Accuracy: 51.4%
  - SOTA MLLMs Accuracy: <30%   


## Dataset Creation

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
