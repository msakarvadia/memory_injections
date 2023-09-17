# Memory Injections:
## Correcting Multi-Hop Reasoning Failures during Inference in Transformer-Based Language Models
This repo contains the code that was used to conduct the experiments in this [paper](https://arxiv.org/abs/2309.05605).

To get a quick introduction to the methods used in this work, checkout this [`demo`](https://colab.research.google.com/drive/1H1jjrdMDRoGj5qRGvAuWuwq1dgIDWjQw?usp=sharing). This demo is also linked under the `demos` folder in this repo.

Answering multi-hop reasoning questions requires retrieving and synthesizing information from diverse sources. Large Language Models (LLMs) struggle to perform such reasoning consistently. Here we propose an approach to pinpoint and rectify multi-hop reasoning failures through targeted memory injections on LLM attention heads. First, we analyze the per-layer activations of GPT-2 models in response to single and multi-hop prompts. We then propose a mechanism that allows users to inject pertinent prompt-specific information, which we refer to as "memories," at critical LLM locations during inference. By thus enabling the LLM to incorporate additional relevant information during inference, we enhance the quality of multi-hop prompt completions. We show empirically that a simple, efficient, and targeted memory injection into a key attention layer can often increase the probability of the desired next token in multi-hop tasks, by up to 424%.

![picture](https://drive.google.com/uc?export=view&id=11PXMPvywR_ZtQNLM615-KB7ltfc0yivM)

## Installation

Requirements: 
`python >=3.7,<3.11`

```
git clone https://github.com/msakarvadia/memory_injections.git
cd memory_injections
conda create --name env python==3.10
conda activate env
pip install -r requirements.txt
```
## Citation

Please cite this work as:



```
@article{sakarvadia2023memory,
  title={Memory Injections: Correcting Multi-Hop Reasoning Failures during Inference in Transformer-Based Language Models},
  author={Sakarvadia, Mansi and Ajith, Aswathy and Khan, Arham and Grzenda, Daniel and Hudson, Nathaniel and Bauer, Andr{\'e} and Chard, Kyle and Foster, Ian},
  journal={arXiv preprint arXiv:2309.05605},
  year={2023}
}
```

